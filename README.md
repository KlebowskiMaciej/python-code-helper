# Gemini Code Helper ( gemini 2.5 pro )
Gemini Code Helper is an interactive Python-based assistant that sends queries to the Google Gemini generative AI API via the command line. 
The application features a dynamic token usage counter displayed in the bottom toolbar (powered by prompt_toolkit) and supports various commands for managing conversation history and adding extra context.

## Features
### Interactive Session: 
Send queries to the Gemini AI model in real time via a command-line interface.

### Dynamic Token Usage Counter:
The token count (a rough estimate based on text length) is updated live in the bottom toolbar while you type your prompt.

### Conversation History:
The conversation history is saved to a file (~/.gemini_conversation_history.txt), allowing you to maintain and continue your discussion between sessions.
If the token usage of the history exceeds 1,000,000 tokens (estimated), you will be warned to reset the history.

### Global Context:
You can include extra context (such as the content of a file or directory) to be appended to every query.

### Built-in Commands:
Within the interactive session you can use the following commands:

**/exit** – quit the session.

**/reset** – clear the conversation history.

**/save-reset** – Save the current conversation history to a backup file (with a timestamp in its name) in your home directory and then clear the conversation history.

**/history** – show conversation statistics (number of tokens and lines).

**/context <text>** – add additional context to the current conversation history

## Installation
### Get Api Key
If you have not set up your API key yet, visit https://aistudio.google.com/apikey to create your dedicated API key.

### Configuration
Before using Gemini Code Helper, you need to provide your Gemini API key. You can configure this in one of two ways:

#### Global Environment Variables:
Set the variables **GEMINI_API_KEY**, **GEMINI_MODEL**, and **MAX_TOKENS** in your system’s environment.

#### Using a .env File:
The first time the tool runs, it checks for a configuration file (e.g. .env) in your home directory. 
If it isn’t present, the tool will prompt you for your API key and save it automatically in the home directory (alongside the conversation history file). 
On subsequent runs, the tool will load the API key from this file.

### Install package
```python
pip install path/to/your_package.whl
```

## Usage

### Running the Interactive Session
To start an interactive session, run:
```pwsh
gemini-code-helper
```
If you want to include a global context (e.g., load a file or directory contents), use:
```pwsh
gemini-code-helper --context "/path/to/context/folder_or_file"
```
To see detailed logging for troubleshooting, add the --verbose flag:
```pwsh
gemini-code-helper --verbose
```
