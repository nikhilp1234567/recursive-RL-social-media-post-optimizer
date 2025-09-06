import json

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
