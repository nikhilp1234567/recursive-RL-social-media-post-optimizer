import json
from helpers.GRPO_Runpod import train_and_generate_post
from helpers.twitter_helpers import post_to_x
from helpers.reinforcement_learning_helpers import append_to_dataset, update_post_metrics, extract_response_from_generation_robust, DATASET_FILE

if __name__ == "__main__":
    print("--- Starting De Novo RL Workflow --- \n")
    try:
        # step 1: grab dataset and update metrics from yesterday's post
        dataset = []
        with open(DATASET_FILE, 'r') as f:
            for line in f:
                if line.strip(): 
                    dataset.append(json.loads(line.strip()))
        
        print(f"Dataset loaded from {DATASET_FILE}. Contains {len(dataset)} entries.")

        # Step 1: Try to update metrics from the last post (non-fatal)
        try:
            last_tweet_id = update_post_metrics(dataset)
            print(f"Previous post metrics grabbed and updated, id: {last_tweet_id}")
        except Exception as e:
            print(f"Warning: Failed to update previous post metrics: {e}. Continuing without updating metrics.")
            last_tweet_id = None

        # Step 2: Train the model and generate a new post
        print("Training model and generating new post...")
        custom_prompt = "You are the social media manager for the twitter account of Panoptic,  a company focussed on modelling the ecosystem services of nature using transformer based foundation models, to highlight the return on investment of nature based infastructure for climate risk mitigation and adaptation. Produce an engaging post, ensuring you adhere to twitter's content guidelines. Keep it extremely short and humanisitic. stay within the twitter length guidelines."
        try:
            generated_response = train_and_generate_post(
                dataset_path=DATASET_FILE, 
                custom_prompt=custom_prompt,
                use_reward_model=True
            )
            
            # Step 3: Extract the actual post content from the generated response
            new_post = extract_response_from_generation_robust(generated_response, custom_prompt)
            
            if new_post and new_post.strip():
                print("\n--- New Post Generated! ---")
                print("=============================")
                print(new_post)
                print("=============================")
                
                # Step 3b: Post the new post to X (non-fatal)
                tweet_id = "00000"  # Default value in case posting fails
                try:
                    tweet_id = post_to_x(new_post)
                    print(f"Posted to X successfully. tweet_id={tweet_id}")
                except Exception as e:
                    print(f"Warning: failed to post to X: {e}. Continuing without posting.")

                # Step 4: Add the new post to the dataset with default metrics
                append_to_dataset(new_post, prompt=custom_prompt, tweet_id=tweet_id)
                
                print("\nPost added to dataset with tweet_id and initial metrics (0 views, 0 likes, 0 reposts).")
                
            else:
                print("Error: Generated response is empty or invalid.")
                print("Full generated response:", generated_response)

        except Exception as e:
            print(f"Error during training and generation: {e}")

    except FileNotFoundError:
        print(f"Error: The file '{DATASET_FILE}' was not found.")
        print("Please ensure you have created this JSONL file in the same directory as the script.")
    except json.JSONDecodeError as e:
        print(f"Error: Could not parse '{DATASET_FILE}'. Line: {e}")
        print("Please ensure the file is a valid JSONL file (one JSON object per line).")
    
    print("\n--- Workflow Complete ---")

