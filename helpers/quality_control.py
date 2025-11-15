from reinforcement_learning_helpers import DATASET_FILE
import json

def check_tweet_length(post:str) -> int:
    if len(post) > 280:
        return 0
    return 1

def check_duplication(current_post:str) -> int:
    with open(DATASET_FILE, 'r') as f:
        for line in f:
            if line.strip(): 
                dataset.append(json.loads(line.strip()))
    for post in dataset:
        if post['post'] == current_post:
            return 0
    return 1

def validate_tweet_content():
    return