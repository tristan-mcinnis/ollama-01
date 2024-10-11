import json
import time
import requests
import yaml
import logging
import sys
from typing import List, Dict, Tuple, Optional, Generator
import re
import csv  # Ensure this import is only here at the top

# Set up logging with RotatingFileHandler to prevent log file from becoming too large
from logging.handlers import RotatingFileHandler

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Stream Handler
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Rotating File Handler
file_handler = RotatingFileHandler("agentic_ollama.log", maxBytes=5*1024*1024, backupCount=5)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Load configuration from YAML file
CONFIG_PATH = 'config.yaml'

def load_config(path: str) -> Dict[str, str]:
    """
    Loads configuration from the specified YAML file.

    Parameters:
        path (str): Path to the YAML configuration file.

    Returns:
        Dict[str, str]: Configuration parameters.
    """
    try:
        with open(path, 'r') as config_file:
            config = yaml.safe_load(config_file)
            logger.info(f"Configuration loaded from {path}")
            return config
    except FileNotFoundError:
        logger.error(f"Configuration file {path} not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {str(e)}")
        sys.exit(1)

config = load_config(CONFIG_PATH)

OLLAMA_URL = config.get('ollama_url', 'http://localhost:11434')
OLLAMA_MODEL = config.get('ollama_model', 'llama3.2')
SYSTEM_PROMPT = config.get('system_prompt', '')

logger.info(f"Ollama URL: {OLLAMA_URL}")
logger.info(f"Ollama Model: {OLLAMA_MODEL}")
logger.info(f"Loaded System Prompt: {SYSTEM_PROMPT}")  # For debugging purposes

def extract_json(content: str) -> Optional[str]:
    """
    Extracts the first JSON object found in the content.

    Parameters:
        content (str): The raw content string from the assistant.

    Returns:
        Optional[str]: The extracted JSON string or None if not found.
    """
    try:
        # This regex finds the first JSON object in the content
        json_pattern = re.compile(r'(\{.*\})', re.DOTALL)
        match = json_pattern.search(content)
        if match:
            return match.group(1)
    except Exception as e:
        logger.error(f"Error during JSON extraction: {str(e)}")
    return None

def parse_json_response(content: str) -> Dict:
    """
    Parses a JSON response from the assistant.

    Parameters:
        content (str): The raw content string from the assistant.

    Returns:
        Dict: Parsed JSON object or an error object.
    """
    json_str = extract_json(content)
    if not json_str:
        logger.warning("No JSON block found in the response.")
        logger.debug(f"Full response content: {content}")
        return {
            "title": "Error",
            "content": "Invalid JSON response format from the assistant.",
            "next_action": "final_answer"
        }
    try:
        parsed_json = json.loads(json_str)
        if validate_response(parsed_json):
            return parsed_json
        else:
            logger.warning("Invalid JSON structure received.")
            return {
                "title": "Error",
                "content": "Assistant provided an invalid response format.",
                "next_action": "final_answer"
            }
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON response: {str(e)}")
        logger.debug(f"Problematic JSON string: {json_str}")
        return {
            "title": "Error",
            "content": "Invalid JSON syntax in the assistant's response.",
            "next_action": "final_answer"
        }

def validate_response(parsed_json: Dict) -> bool:
    """
    Validates the structure and content of the parsed JSON response.

    Parameters:
        parsed_json (Dict): The parsed JSON object.

    Returns:
        bool: True if valid, False otherwise.
    """
    required_keys = {'title', 'content', 'next_action'}
    if not required_keys.issubset(parsed_json.keys()):
        return False
    if not isinstance(parsed_json['title'], str):
        return False
    if not isinstance(parsed_json['content'], str):
        return False
    if parsed_json['next_action'] not in {'continue', 'final_answer'}:
        return False
    return True

def make_api_call(messages: List[Dict[str, str]], max_tokens: int, is_final_answer: bool = False) -> Dict:
    """
    Makes an API call to the Ollama model with retry and exponential backoff.

    Parameters:
        messages (List[Dict[str, str]]): Conversation messages.
        max_tokens (int): Maximum number of tokens to generate.
        is_final_answer (bool): Flag indicating if this call is for the final answer.

    Returns:
        Dict: The parsed JSON response from the assistant or an error object.
    """
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Attempting API call (Attempt {attempt})")
            response = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": messages,
                    "stream": False,
                    "format": "json",  # Enable JSON mode
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.2  # Adjust for some randomness
                    }
                },
                timeout=60  # Set a timeout for the request
            )
            response.raise_for_status()
            response_json = response.json()
            content = response_json.get("message", {}).get("content", "").strip()
            logger.info(f"API call successful. Response content length: {len(content)} characters")

            if not content:
                logger.warning("Empty content received from assistant.")
                return {
                    "title": "Error",
                    "content": "Received empty response from the assistant.",
                    "next_action": "final_answer" if is_final_answer else "continue"
                }

            parsed_json = parse_json_response(content)
            return parsed_json

        except requests.exceptions.RequestException as e:
            logger.error(f"API call failed on attempt {attempt}: {str(e)}")
            if attempt < max_retries:
                backoff_time = 2 ** (attempt - 1)
                logger.info(f"Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)
            else:
                error_message = f"Failed to generate {'final answer' if is_final_answer else 'step'} after {max_retries} attempts. Error: {str(e)}"
                logger.error(error_message)
                return {
                    "title": "Error",
                    "content": error_message,
                    "next_action": "final_answer" if not is_final_answer else "final_answer"
                }

def generate_response(prompt: str, max_steps: int = 10) -> Generator[Tuple[List[Tuple[str, str, float]], Optional[float]], None, None]:
    """
    Generates a step-by-step response by interacting with the Ollama model.

    Parameters:
        prompt (str): The user's query.
        max_steps (int): Maximum number of reasoning steps.

    Yields:
        Generator[Tuple[List[Tuple[str, str, float]], Optional[float]], None, None]: A list of steps and total thinking time.
    """
    logger.info(f"Generating response for prompt: {prompt}")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "Thank you! I will now think step by step following my instructions, starting at the beginning after decomposing the problem."}
    ]

    steps: List[Tuple[str, str, float]] = []
    step_count = 1
    total_thinking_time = 0.0

    while step_count <= max_steps:
        start_time = time.time()
        step_data = make_api_call(messages, max_tokens=300)
        end_time = time.time()
        thinking_time = end_time - start_time
        total_thinking_time += thinking_time

        logger.info(f"Step {step_count} completed in {thinking_time:.2f} seconds")
        logger.debug(f"Step {step_count} data: {step_data}")

        # Ensure step_data has the required keys
        if 'title' not in step_data:
            step_data['title'] = f"Step {step_count}"
        if 'content' not in step_data:
            step_data['content'] = str(step_data)
        if 'next_action' not in step_data:
            step_data['next_action'] = 'continue'

        # Handle error responses
        if step_data.get('title') == 'Error':
            steps.append((f"Step {step_count}: {step_data['title']}", step_data['content'], thinking_time))
            yield steps, total_thinking_time
            return  # Exit on error

        steps.append((f"Step {step_count}: {step_data['title']}", step_data['content'], thinking_time))

        # Append the assistant's step to the messages for context
        messages.append({"role": "assistant", "content": json.dumps(step_data)})

        print(f"Step {step_count} completed")

        if step_data['next_action'] == 'final_answer' or step_count == max_steps:
            logger.info("Final answer reached or maximum steps reached")
            break

        step_count += 1

        yield steps, None  # Yield steps and indicate that total thinking time is not yet finalized

    # Generate final answer
    print("Generating final answer")
    messages.append({"role": "user", "content": "Please provide the final answer based on your reasoning above."})

    start_time = time.time()
    final_data = make_api_call(messages, max_tokens=200, is_final_answer=True)
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time

    logger.info(f"Final answer generated in {thinking_time:.2f} seconds")
    logger.debug(f"Final answer data: {final_data}")

    # Ensure final_data has the required keys
    if 'title' not in final_data:
        final_data['title'] = "Final Answer"
    if 'content' not in final_data:
        final_data['content'] = str(final_data)

    steps.append(("Final Answer", final_data['content'], thinking_time))

    yield steps, total_thinking_time

def save_response(steps: List[Tuple[str, str, float]], total_time: float):
    """
    Saves the response steps and total thinking time to a CSV file.

    Parameters:
        steps (List[Tuple[str, str, float]]): List of steps with title, content, and time.
        total_time (float): Total thinking time in seconds.
    """
    try:
        with open('response_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            for step in steps:
                writer.writerow(step)
            writer.writerow(['Total Thinking Time', total_time])
        logger.info("Response successfully saved to response_log.csv")
    except IOError as e:
        logger.error(f"IOError while saving response to CSV: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error while saving response to CSV: {str(e)}")

def main():
    user_query = input("Enter your query: ")

    if user_query:
        print("Generating response...")

        for steps, total_thinking_time in generate_response(user_query):
            for title, content, thinking_time in steps:
                print(f"\n{title}")
                print(f"Content: {content}")
                print(f"Time: {thinking_time:.2f}s")

            if total_thinking_time is not None:
                print(f"\nTotal thinking time: {total_thinking_time:.2f} seconds")
                save_response(steps, total_thinking_time)

if __name__ == "__main__":
    main()
