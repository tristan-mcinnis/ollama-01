This repository contains a Python script designed to interact with the Ollama Llama3.2 model to generate step-by-step AI-driven reasoning. The assistant provides structured JSON responses for each reasoning phase and offers detailed explanations of its thought process.

## Features

- **Step-by-step Reasoning:** The AI breaks down its thought process, generating individual steps for clear reasoning.
- **Strict JSON Formatting:** The system responds with a well-structured JSON object that includes a title, content, and next action (`continue` or `final_answer`).
- **Error Handling:** Built-in logging and error handling ensure the system can retry failed API calls and manage invalid responses.
- **Configuration via YAML:** Easily configure the Ollama API URL and model via a `config.yaml` file.
- **CSV Logging:** Saves the AI’s responses and thinking time to a CSV file for further analysis.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/agentic-ollama.git
   cd agentic-ollama
Install required dependencies:

bash
Copy code
pip install -r requirements.txt
Create a config.yaml file:

yaml
Copy code
ollama_url: 'http://localhost:11434'
ollama_model: 'llama3.2'
system_prompt: |
  You are an expert AI assistant with advanced reasoning capabilities...

## Usage
Run the script:

```bash
python agentic_ollama.py
```

Input your query when prompted, and the system will generate a step-by-step reasoning response, which will be logged both in the console and in a CSV file (response_log.csv).

## Configuration

   config.yaml: The configuration file allows you to set the Ollama model and API endpoint.
   Logging: Log files are rotated and saved under agentic_ollama.log with detailed information about each API call and any encountered errors.
   Example
   Here is an example of a valid AI-generated response:

json
```bash
   {
     "title": "Step 1: Identifying Characters",
     "content": "First, I will identify all the characters in the word 'strawberry' to accurately count the number of 'r's.",
     "next_action": "continue"
   }
```

## Error Handling

The system handles errors by:

Logging each failed API call attempt.
Automatically retrying failed requests with exponential backoff.
Providing fallback messages in case of invalid or empty responses.
Logging and Response Saving
All responses are logged to the console and saved in response_log.csv with step titles, contents, and thinking times.
