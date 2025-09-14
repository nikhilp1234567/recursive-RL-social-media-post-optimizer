import json
from twitter_functions import fetch_metrics_x

# Path to your GRPO training dataset file
DATASET_FILE = "data.jsonl"

def append_to_dataset(new_post, views=0, likes=0, reposts=0, prompt="failed to gather", tweet_id="00000"):
    """
    Append a new post to the dataset file.
    
    Args:
        new_post (str): The generated post content
        views (int): Number of views (default 0 for new posts)
        likes (int): Number of likes (default 0 for new posts)
        reposts (int): Number of reposts (default 0 for new posts)
        prompt (str): The prompt used to generate the post
        tweet_id (str or int): The tweet ID associated with the post (default 00000)
    """
    new_entry = {
        "prompt": prompt,
        "post": new_post,
        "views": views,
        "likes": likes,
        "reposts": reposts, 
        "tweet_id": tweet_id
    }
    
    # Append to the JSONL file
    with open(DATASET_FILE, 'a') as f:
        f.write('\n' + json.dumps(new_entry))
    
    print(f"New post added to dataset: {DATASET_FILE}")

def update_post_metrics(dataset):
    """
    Update the metrics for the last post in the dataset by fetching current metrics from X.
    
    Args:
        dataset (list): The dataset containing posts with tweet_id fields
    
    Returns:
        str: The tweet_id of the updated post, or None if update failed
    """
    if not dataset:
        print("Dataset is empty. No posts to update.")
        return None
    
    # Get the last entry from the dataset
    last_entry = dataset[-1]
    tweet_id = last_entry.get('tweet_id')
    
    if not tweet_id:
        print("No valid tweet_id found in the last entry.")
        return None
    
    try:
        # Fetch current metrics from X
        print(f"Fetching metrics for tweet_id: {tweet_id}")
        metrics = fetch_metrics_x(str(tweet_id))
        
        # Update the last entry with new metrics
        last_entry['views'] = metrics.get('quotes', 0)  # quotes = impressions/views
        last_entry['likes'] = metrics.get('likes', 0)
        last_entry['reposts'] = metrics.get('reposts', 0)
        
        print(f"Updated metrics - Views: {last_entry['views']}, Likes: {last_entry['likes']}, Reposts: {last_entry['reposts']}")
        
        # Write the updated dataset back to file
        with open(DATASET_FILE, 'w') as f:
            for entry in dataset:
                f.write(json.dumps(entry) + '\n')
        
        return str(tweet_id)
        
    except Exception as e:
        print(f"Error fetching metrics for tweet {tweet_id}: {e}")
        return None

def extract_response_from_generation(generated_text, prompt):
    """
    Extract the actual response from the generated text by removing the prompt.
    """
    # Remove the instruction format and extract just the response
    if "### Response:" in generated_text:
        response_part = generated_text.split("### Response:")[-1].strip()
        
        # Additional cleanup: remove any repeated sections
        lines = response_part.split('\n')
        seen_lines = set()
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            if line and line not in seen_lines:
                seen_lines.add(line)
                clean_lines.append(line)
            elif line == "---":  # Stop at separators
                break
                
        return '\n'.join(clean_lines)
    else:
        # Fallback: try to remove the prompt from the beginning
        if generated_text.startswith(prompt):
            return generated_text[len(prompt):].strip()
        return generated_text.strip()
        
def extract_response_from_generation_robust(generated_text, prompt):
    """
    Robust extraction of the actual response from the generated text by removing 
    the prompt, reasoning, metadata, and other extraneous content.
    
    Args:
        generated_text (str): The full generated text
        prompt (str): The original prompt
    
    Returns:
        str: The extracted clean response
    """
    import re
    
    # Start with the full generated text
    response_part = generated_text
    
    # Keep splitting on Final Answer delimiters until no more exist
    final_answer_patterns = ["### Final Answer:", "**Final Answer:**", "### Response:", "### Answer:"]
    
    for pattern in final_answer_patterns:
        while pattern in response_part:
            response_part = response_part.split(pattern)[-1].strip()
    
    # If we still have the original text, try to remove the prompt from the beginning
    if response_part == generated_text and generated_text.startswith(prompt):
        response_part = generated_text[len(prompt):].strip()
    
    # Split into lines for processing
    lines = response_part.split('\n')
    clean_lines = []
    in_main_content = False
    seen_lines = set()
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines at the beginning
        if not line and not in_main_content:
            continue
            
        # Stop at common separators and metadata sections
        if line.startswith('---') or line.startswith('**Word count:**') or \
           line.startswith('**Hashtags:**') or line.startswith('**Tone:**') or \
           line.startswith('**Emojis:**') or line.startswith('**Call to action:**') or \
           line.startswith('**Compliance') or line.startswith('*(Replace ') or \
           line.startswith('*Short,') or line.startswith('*Note:'):
            break
            
        # Skip reasoning sections (common LLM patterns)
        if any(phrase in line.lower() for phrase in [
            'let me', 'i need to', 'first,', 'next,', 'maybe', 'wait,', 
            'let me check', 'let me count', 'alright,', 'looks good',
            'i should', 'that should work', 'double-check'
        ]):
            continue
            
        # Skip lines that look like internal reasoning
        if line.startswith('Okay,') or line.startswith('That uses') or \
           line.startswith('Including a hashtag') or line.startswith('Maybe include') or \
           line.startswith('Wait,') or line.startswith('Let me'):
            continue
            
        # Remove markdown formatting markers
        line = re.sub(r'^\*\*([^*]+)\*\*$', r'\1', line)  # **text** -> text
        line = re.sub(r'^\*([^*]+)\*$', r'\1', line)      # *text* -> text
        
        # Mark that we've started collecting main content
        if line and not in_main_content:
            in_main_content = True
            
        # Add unique non-empty lines
        if line and line not in seen_lines:
            seen_lines.add(line)
            clean_lines.append(line)
    
    # Join the clean lines
    result = '\n'.join(clean_lines)
    
    # Additional cleanup: remove common prefixes/suffixes
    result = re.sub(r'^(Here\'s|Here is|This is)\s+', '', result, flags=re.IGNORECASE)
    result = re.sub(r'\s*\(Replace.*?\).*$', '', result, flags=re.MULTILINE)
    
    # Clean up extra whitespace
    result = re.sub(r'\n\s*\n', '\n', result)  # Multiple newlines -> single
    result = result.strip()
    
    return result

def extract_response_from_generation_old(generated_text, prompt):
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
