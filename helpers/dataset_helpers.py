import json
import os


def add_to_dataset(prompt: str, post: str, dataset_path: str = "data_africfood/dataset.jsonl") -> None:
    """
    Add a new entry to the dataset.jsonl file.
    
    Args:
        prompt: The prompt text
        post: The post text
        dataset_path: Path to the dataset.jsonl file (default: /workspace/RL_AI/data/dataset.jsonl)
    """
    entry = {
        "prompt": prompt,
        "post": post,
        "views": 0,
        "likes": 0,
        "reposts": 0,
        "tweet_id": "0",
        "posted": False
    }
    
    # Append to the dataset file
    with open(dataset_path, 'a') as f:
        f.write(json.dumps(entry) + '\n')
