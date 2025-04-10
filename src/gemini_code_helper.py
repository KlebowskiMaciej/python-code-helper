
import os
import sys
import time
import logging
import argparse
import threading
import itertools
import datetime
import google.generativeai as genai
from pathlib import Path
from typing import Optional, Generator
from dotenv import load_dotenv
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.output.win32 import NoConsoleScreenBufferError
except ImportError:
    PromptSession = None

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

HISTORY_FILE = os.path.join(os.path.expanduser("~"), ".gemini_conversation_history.txt")

max_tokens = int(os.getenv("MAX_TOKENS", "900000"))

def spinner_task(stop_event):
    spinner = itertools.cycle(["|", "/", "-", "\\"])
    while not stop_event.is_set():
        sys.stdout.write("\rLoading " + next(spinner))
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\r" + " " * 20 + "\r")
    sys.stdout.flush()

class GeminiCodeHelper:
    _ROUGH_CHARS_PER_TOKEN = 4

    def __init__(self, logger: logging.Logger, gemini_model: str, api_key: str, max_tokens: int):
        if not isinstance(logger, logging.Logger):
            raise TypeError("Logger must be an instance of logging.Logger")
        if not gemini_model or not isinstance(gemini_model, str):
            raise ValueError("Gemini model must be provided as a string")
        if not api_key or not isinstance(api_key, str):
            raise ValueError("API key is required and must be a string")
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        self._logger = logger
        self._gemini_model = gemini_model
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._model = None
        try:
            genai.configure(api_key=self._api_key)
            self._logger.info("Successfully configured Google Generative AI with the provided API key.")
        except Exception as e:
            self._logger.error(f"Error during Google Generative AI configuration: {e}", exc_info=True)

    def _initialize_model(self) -> None:
        if self._model is None:
            try:
                self._model = genai.GenerativeModel(self._gemini_model)
                self._logger.info(f"Gemini model '{self._gemini_model}' initialized successfully.")
            except Exception as e:
                self._logger.error(f"Failed to initialize Gemini model: {e}", exc_info=True)
                raise RuntimeError("Error initializing Gemini model.") from e

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // self._ROUGH_CHARS_PER_TOKEN

    def _read_file_content(self, file_path: Path) -> Optional[str]:
        try:
            return file_path.read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            self._logger.error(f"Error reading file {file_path}: {e}", exc_info=True)
        return None

    def _walk_directory(self, dir_path: Path) -> Generator[Path, None, None]:
        for item in dir_path.rglob('*'):
            if item.is_file():
                if any(ignored in item.parts for ignored in ['.git', 'node_modules', 'venv', '__pycache__']):
                    continue
                yield item
            elif item.is_dir():
                yield from self._walk_directory(item)

    def _load_directory_content(self, dir_path: Path) -> str:
        contents = []
        for file_path in self._walk_directory(dir_path):
            file_content = self._read_file_content(file_path)
            if file_content:
                contents.append(file_content)
        self._logger.info("Finished loading directory content.")
        return "\n".join(contents)

    def load_context_from_path(self, path_str: str, history_tokens: int) -> Optional[str]:
        """
        Load context from file or directory.
        Maximum allowed context tokens = max_tokens - history_tokens - 20000.
        If the loaded context exceeds the allowed token count, return None.
        """
        context_path = Path(path_str)
        self._logger.info(f"Attempting to load context from path: {context_path}")
        if not context_path.exists():
            self._logger.error(f"Path does not exist: {context_path}")
            return None
        if context_path.is_file():
            loaded_context = self._read_file_content(context_path)
        elif context_path.is_dir():
            loaded_context = self._load_directory_content(context_path)
        else:
            self._logger.error(f"Path is neither a file nor a directory: {context_path}")
            return None

        if loaded_context:
            available_context_tokens = self._max_tokens - history_tokens - 20000
            if available_context_tokens < 0:
                available_context_tokens = 0
            if self._estimate_tokens(loaded_context) > available_context_tokens:
                self._logger.error("The provided context exceeds the available token limit."
                                   " Please reset the conversation or provide a smaller context.")
                return None
        return loaded_context

    def ask_gemini(self, prompt: str, context_path: Optional[str] = None, history: str = "") -> str:
        self._initialize_model()
        full_prompt = ("This is the conversation history:\n" + history +
                       "\n--- End of conversation history ---\n" +
                       "Now the current question:\n" + prompt)
        if context_path:
            history_tokens = self._estimate_tokens(history)
            loaded_context = self.load_context_from_path(context_path, history_tokens)
            if loaded_context is None:
                return ("ERROR: The provided context exceeds the allowed token limit. "
                        "Please reset the conversation using '/reset' or '/save-reset' and try again with a smaller context.")
            self._logger.info("Global context loaded successfully.")
            full_prompt += f"\n\n--- Global Context ---\n{loaded_context}\n--- End of Global Context ---"
        total_tokens = self._estimate_tokens(full_prompt)
        if total_tokens >= self._max_tokens:
            return (f"ERROR: The token count for the conversation history, context, and prompt ({total_tokens} tokens) "
                    f"exceeds the maximum allowed ({self._max_tokens} tokens). Please reset the conversation.")
        self._logger.info("Sending request to Gemini model.")
        stop_spinner = threading.Event()
        spinner_thread = threading.Thread(target=spinner_task, args=(stop_spinner,))
        spinner_thread.start()
        try:
            response = self._model.generate_content(full_prompt)
        except Exception as e:
            stop_spinner.set()
            spinner_thread.join()
            self._logger.error(f"Error during communication with Gemini API: {e}", exc_info=True)
            return f"ERROR: Communication with Gemini API failed: {e}"
        stop_spinner.set()
        spinner_thread.join()
        if not getattr(response, 'parts', None):
            if getattr(response, 'prompt_feedback', None) and getattr(response.prompt_feedback, 'block_reason', None):
                reason = response.prompt_feedback.block_reason.name
                self._logger.error(f"Response blocked by Gemini safety filters. Reason: {reason}")
                return f"ERROR: Response blocked by Gemini safety filters (Reason: {reason}). Please rephrase your prompt."
            else:
                self._logger.warning("Gemini returned an empty response without a specified block reason.")
                return "INFO: Gemini returned an empty response."
        self._logger.info("Response received from Gemini model.")
        return response.text

conv_history = ""

def load_conversation_history() -> str:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"Error reading conversation history: {e}", file=sys.stderr)
            return ""
    return ""

def save_conversation_history(history: str) -> None:
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            f.write(history)
    except Exception as e:
        print(f"Error writing conversation history: {e}", file=sys.stderr)

def estimate_tokens(text: str) -> int:
    return len(text) // 4

def print_history_stats(history: str) -> None:
    tokens = estimate_tokens(history)
    lines = history.count("\n")
    print(f"[History stats] Tokens: {tokens}, Lines: {lines}")

def interactive_session(helper: GeminiCodeHelper, initial_history: str, global_context: Optional[str]) -> None:
    global conv_history
    conv_history = initial_history

    def bottom_toolbar():
        current_input = session.default_buffer.document.text
        total_tokens = estimate_tokens(conv_history + current_input)
        return f"Tokens used: {total_tokens}/{max_tokens}"

    using_prompt_toolkit = True
    try:
        session = PromptSession(bottom_toolbar=bottom_toolbar)
    except Exception as e:
        from prompt_toolkit.output.win32 import NoConsoleScreenBufferError
        if isinstance(e, NoConsoleScreenBufferError):
            print("No Windows console found. Falling back to standard input mode.")
            using_prompt_toolkit = False
        else:
            raise e

    print("Interactive session with Gemini. Type your prompt below.")
    print("Commands:")
    print("  /exit           - quit")
    print("  /reset          - clear history")
    print("  /save-reset     - save history to backup file and reset conversation")
    print("  /history        - show conversation stats")
    print("  /context <text> - add additional context\n")

    while True:
        if using_prompt_toolkit:
            try:
                user_input = session.prompt("Enter your prompt: ").strip()
            except KeyboardInterrupt:
                print("\nInteractive session terminated by user.")
                break
        else:
            current_tokens = estimate_tokens(conv_history)
            prompt_message = f"Enter your prompt ({current_tokens}/{max_tokens} tokens used): "
            try:
                user_input = input(prompt_message).strip()
            except KeyboardInterrupt:
                print("\nInteractive session terminated by user.")
                break

        if user_input.lower() in {"exit", "/exit", "quit", "/quit"}:
            print("Exiting interactive session.")
            break
        elif user_input.lower() == "/reset":
            conv_history = ""
            save_conversation_history(conv_history)
            print("Conversation history has been reset.")
            continue
        elif user_input.lower() == "/save-reset":
            backup_filename = os.path.join(os.path.expanduser("~"),
                                           f".gemini_history_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            try:
                with open(backup_filename, "w", encoding="utf-8") as f:
                    f.write(conv_history)
                print(f"Conversation history saved to: {backup_filename}")
            except Exception as e:
                print(f"Error writing backup history file: {e}", file=sys.stderr)
            conv_history = ""
            save_conversation_history(conv_history)
            print("Conversation history has been reset.")
            continue
        elif user_input.lower() == "/history":
            print_history_stats(conv_history)
            continue
        elif user_input.lower().startswith("/context"):
            extra_context = user_input[len("/context"):].strip()
            if extra_context:
                conv_history += f"\n--- Additional Context ---\n{extra_context}\n--- End of Additional Context ---\n"
                save_conversation_history(conv_history)
                print("Additional context has been added to the conversation history.")
            else:
                print("Usage: /context <your context text>")
            continue

        full_prompt = ("This is the conversation history:\n" + conv_history +
                       "\n--- End of conversation history ---\n" +
                       "Now the current question:\n" +
                       f"User: {user_input}\n")
        if global_context:
            full_prompt += f"\n--- Global Context ---\n{global_context}\n--- End of Global Context ---\n"
        total_tokens = estimate_tokens(full_prompt)
        if total_tokens >= max_tokens:
            print(f"WARNING: The token count for the conversation history and prompt ({total_tokens} tokens) exceeds our safe limit ({max_tokens} tokens).")
            print("Please reset the conversation using '/reset' or '/save-reset' and start over.")
            continue

        print("Sending request...")
        response = helper.ask_gemini(full_prompt, context_path=None, history=conv_history)
        print("\nGemini Response:")
        print(response)
        new_history_entry = f"User: {user_input}\nGemini: {response}\n"
        conv_history += new_history_entry
        if estimate_tokens(conv_history) >= max_tokens:
            print("WARNING: Conversation history exceeds the maximum allowed tokens. Use '/reset' to clear history.")
        else:
            save_conversation_history(conv_history)

def main():
    parser = argparse.ArgumentParser(
        description="Interactive session with the Gemini model using persistent conversation history."
    )
    parser.add_argument("--context", type=str, help="Path to file or directory for additional context.", default=None)
    parser.add_argument("--reset", action="store_true", help="Reset the conversation history on startup.")
    parser.add_argument("--verbose", action="store_true", help="Display detailed logger messages.")
    args = parser.parse_args()


    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("GeminiCodeHelperApp")
    if args.verbose:
        stream_handler = logging.StreamHandler(stream=sys.stderr)
        stream_handler.setFormatter(log_formatter)
        logger.addHandler(stream_handler)
        logger.setLevel(logging.INFO)
    else:
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.CRITICAL)

    global_env_path = os.path.join(os.path.expanduser("~"), ".gemini_config.env")
    if not os.path.exists(global_env_path) or "GEMINI_API_KEY" not in os.environ:
        api_key = input("Please enter your Gemini API Key (visit https://aistudio.google.com/apikey to generate one): ").strip()
        with open(global_env_path, "w", encoding="utf-8") as f:
            f.write(f"GEMINI_API_KEY={api_key}\n")
        os.environ["GEMINI_API_KEY"] = api_key
    else:
        load_dotenv(dotenv_path=global_env_path)

    if args.reset and os.path.exists(HISTORY_FILE):
        try:
            os.remove(HISTORY_FILE)
            logger.info("Conversation history has been reset as requested.")
        except Exception as e:
            logger.error(f"Error resetting conversation history: {e}", exc_info=True)

    conversation_history = load_conversation_history()
    if estimate_tokens(conversation_history) >= max_tokens:
        print("ERROR: Conversation history exceeds the maximum allowed tokens. Please reset the conversation using the /reset command.", file=sys.stderr)
        return

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        logger.error("Gemini API key not found in environment variables (GEMINI_API_KEY). Exiting.")
        return

    gemini_model = "gemini-2.5-pro-exp-03-25"
    max_tokens_env = int(os.getenv("MAX_TOKENS", str(max_tokens)))

    helper = GeminiCodeHelper(logger=logger, gemini_model=gemini_model, api_key=gemini_api_key, max_tokens=max_tokens_env)

    global_context = None
    if args.context:
        context_path = Path(args.context)
        if context_path.exists():
            if context_path.is_file():
                with open(context_path, "r", encoding="utf-8", errors="ignore") as f:
                    global_context = f.read()
            elif context_path.is_dir():
                global_context = helper.load_context_from_path(args.context, estimate_tokens(conversation_history))
        else:
            print(f"WARNING: Provided context path '{args.context}' does not exist.", file=sys.stderr)

    interactive_session(helper, conversation_history, global_context)
