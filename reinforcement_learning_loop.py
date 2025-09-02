import json
import requests 
import time
from GRPO_Runpod import train_and_generate_post
from twitter_functions import post_to_x   

# Path to your GRPO training dataset file
DATASET_FILE = "data.jsonl"

def append_to_dataset(new_post, views=0, likes=0, reposts=0, prompt="failed to gather"):
    """
    Append a new post to the dataset file.
    
    Args:
        new_post (str): The generated post content
        views (int): Number of views (default 0 for new posts)
        likes (int): Number of likes (default 0 for new posts)
        reposts (int): Number of reposts (default 0 for new posts)
        prompt (str): The prompt used to generate the post
    """
    new_entry = {
        "prompt": prompt,
        "post": new_post,
        "views": views,
        "likes": likes,
        "reposts": reposts
    }
    
    # Append to the JSONL file
    with open(DATASET_FILE, 'a') as f:
        f.write(json.dumps(new_entry) + '\n')
    
    print(f"New post added to dataset: {DATASET_FILE}")

def update_post_metrics(post_content, views, likes, reposts):
    """
    Update the metrics for an existing post in the dataset.
    
    Args:
        post_content (str): The content of the post to update
        views (int): Updated number of views
        likes (int): Updated number of likes
        reposts (int): Updated number of reposts
    
    Returns:
        bool: True if post was found and updated, False otherwise
    """
    updated = False
    dataset = []
    
    # Read all entries
    with open(DATASET_FILE, 'r') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line.strip())
                if entry['post'] == post_content:
                    entry['views'] = views
                    entry['likes'] = likes
                    entry['reposts'] = reposts
                    updated = True
                    print(f"Updated metrics for post: {post_content[:50]}...")
                dataset.append(entry)
    
    if updated:
        # Rewrite the file with updated data
        with open(DATASET_FILE, 'w') as f:
            for entry in dataset:
                f.write(json.dumps(entry) + '\n')
        print("Dataset file updated successfully.")
    else:
        print("Post not found in dataset. No updates made.")
    
    return updated

def extract_response_from_generation(generated_text, prompt):
    """
    Extract the actual response from the generated text by removing the prompt.
    
    Args:
        generated_text (str): The full generated text
        prompt (str): The original prompt
    
    Returns:
        str: The extracted response
    """
    # Remove the instruction format and extract just the response
    if "### Response:" in generated_text:
        response_part = generated_text.split("### Response:")[-1].strip()
        return response_part
    else:
        # Fallback: try to remove the prompt from the beginning
        if generated_text.startswith(prompt):
            return generated_text[len(prompt):].strip()
        return generated_text.strip()

# --- Main Script Logic ---
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

def interactive_metric_update():
    """
    Interactive function to update metrics for existing posts.
    """
    print("\n--- Update Post Metrics ---")
    
    # Load and display recent posts
    dataset = []
    with open(DATASET_FILE, 'r') as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line.strip()))
    
    if not dataset:
        print("No posts found in dataset.")
        return
    
    print("Recent posts:")
    for i, entry in enumerate(dataset[-5:], 1):  # Show last 5 posts
        print(f"{i}. {entry['post'][:100]}...")
        print(f"   Current metrics - Views: {entry['views']}, Likes: {entry['likes']}, Reposts: {entry['reposts']}")
    
    try:
        choice = int(input(f"\nEnter post number to update (1-{min(5, len(dataset))}): ")) - 1
        if choice < 0 or choice >= min(5, len(dataset)):
            print("Invalid choice.")
            return
        
        selected_post = dataset[-(5-choice)]
        
        print(f"\nSelected post: {selected_post['post']}")
        
        views = int(input("Enter new views count: "))
        likes = int(input("Enter new likes count: "))
        reposts = int(input("Enter new reposts count: "))
        
        update_post_metrics(selected_post['post'], views, likes, reposts)
        
    except (ValueError, IndexError) as e:
        print(f"Error: {e}")
        print("Please enter valid numbers.")

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
