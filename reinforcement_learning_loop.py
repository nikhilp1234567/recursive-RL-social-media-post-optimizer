import json
import requests 
import time
from GRPO_Runpod import train_and_generate_post
from twitter_functions import post_to_x
from helper_functions import append_to_dataset, update_post_metrics, extract_response_from_generation, DATASET_FILE, interactive_metric_update

def run_rl_workflow():
    """
    Handles the entire de novo reinforcement learning workflow.
    It reads the local dataset, trains both the reward model and GRPO model,
    and generates a new post that gets added to the dataset.
    """
    print("--- Starting De Novo RL Workflow ---")

    try:
        # Step 1: Read the GRPO training dataset
        print(f"Loading dataset from '{DATASET_FILE}'...")
        dataset = []
        with open(DATASET_FILE, 'r') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    dataset.append(json.loads(line.strip()))
        
        print(f"Dataset loaded. Contains {len(dataset)} entries.")

        # Step 2: Train the model and generate a new post
        print("Training model and generating new post...")
        custom_prompt = "You are the social media post generation engine for the twitter account of a company focussed on modelling the ecosystem services of nature, to highlight the return on investment of nature based infastructure for climate risk mitigation and adaptation. Produce an engaging post, ensuring you adhere to twitter's content guidelines."
        
        try:
            # This returns the generated text directly, not an HTTP response
            generated_response = train_and_generate_post(
                dataset_path=DATASET_FILE, 
                custom_prompt=custom_prompt,
                use_reward_model=True
            )
            
            # Step 3: Extract the actual post content from the generated response
            new_post = extract_response_from_generation(generated_response, custom_prompt)
            
            if new_post and new_post.strip():
                print("\n--- New Post Generated! ---")
                print("=============================")
                print(new_post)
                print("=============================")
                
                # Step 3b: Post the new post to X (non-fatal)
                try:
                    tweet_id = post_to_x(new_post)
                    print(f"Posted to X successfully. tweet_id={tweet_id}")
                except Exception as e:
                    print(f"Warning: failed to post to X: {e}. Continuing without posting.")

                # Step 4: Add the new post to the dataset with default metrics
                append_to_dataset(new_post, prompt=custom_prompt)
                
                print("\nPost added to dataset with initial metrics (0 views, 0 likes, 0 reposts).")
                print("After posting to social media, update the metrics in the dataset file.")
                print("Then run this script again to continue the reinforcement learning loop.")
                
                return new_post
            else:
                print("Error: Generated response is empty or invalid.")
                print("Full generated response:", generated_response)
                return None

        except Exception as e:
            print(f"Error during training and generation: {e}")
            return None

    except FileNotFoundError:
        print(f"Error: The file '{DATASET_FILE}' was not found.")
        print("Please ensure you have created this JSONL file in the same directory as the script.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Could not parse '{DATASET_FILE}'. Line: {e}")
        print("Please ensure the file is a valid JSONL file (one JSON object per line).")
        return None
    
    print("\n--- Workflow Complete ---")

def main():
    """
    Main function that provides options for running the workflow.
    """
    print("=== Reinforcement Learning Workflow ===")
    print("1. Generate new post (full RL workflow)")
    print("2. Update metrics for existing post")
    print("3. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == "1":
                run_rl_workflow()
                break
            elif choice == "2":
                interactive_metric_update()
                break
            elif choice == "3":
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main()
