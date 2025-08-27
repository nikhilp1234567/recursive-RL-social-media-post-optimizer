import json
import requests
import time
from GRPO_Runpod import train_and_generate_post


# Path to your GRPO training dataset file
DATASET_FILE = "data.jsonl"

# --- Main Script Logic ---
def run_rl_workflow():
    """
    Handles the entire de novo reinforcement learning workflow.
    It reads the local dataset, sends it to the LLM for training,
    and displays the newly generated post.
    """
    print("--- Starting De Novo RL Workflow ---")

    try:
        # Step 1: Read the GRPO training dataset
        print(f"Loading dataset from '{DATASET_FILE}'...")
        with open(DATASET_FILE, 'r') as f:
            dataset = json.load(f)
        
        print(f"Dataset loaded. Contains {len(dataset)} entries.")

        # Step 2: Send the dataset to the LLM API for training and generation
        print(f"Sending dataset to the LLM...")
        
        # Use a try-except block to handle potential network or API errors
        try:
            response = train_and_generate_post(dataset_path=DATASET_FILE, custom_prompt="Write a poem about AI");
            response.raise_for_status()  # Raise an exception for bad status codes
            
            # Step 3: Parse the response and get the new post content
            response_data = response.json()
            new_post = response_data.get("new_post")
            
            if new_post:
                print("\n--- New Post Generated! ---")
                print("=============================")
                print(new_post)
                print("=============================")
                print("\nCopy this text and post it to Twitter and Bluesky.")
                print("Remember to collect new metrics in 24 hours and update your 'data.json' file.")
            else:
                print("Error: API response did not contain the 'new_post' field.")
                print("Full response:", response_data)

        except requests.exceptions.RequestException as e:
            print(f"Error connecting to the API: {e}")
            print("Please check your RUNPOD_API_URL and ensure the service is running.")
            return

    except FileNotFoundError:
        print(f"Error: The file '{DATASET_FILE}' was not found.")
        print("Please ensure you have created this JSON file in the same directory as the script.")
    except json.JSONDecodeError:
        print(f"Error: Could not parse '{DATASET_FILE}'.")
        print("Please ensure the file is a valid JSON array.")
    
    print("\n--- Workflow Complete ---")

if __name__ == "__main__":
    run_rl_workflow()
