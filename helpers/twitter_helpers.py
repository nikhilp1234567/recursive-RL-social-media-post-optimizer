import tweepy
import os
import json
from dotenv import load_dotenv
load_dotenv()

def get_x_client():
    """
    Creates and returns a Tweepy Client for X (formerly Twitter) API.

    Returns:
        tweepy.Client: An authenticated Tweepy Client instance.
    """

    consumer_key = os.getenv("X_CONSUMER_KEY")
    consumer_secret = os.getenv("X_CONSUMER_SECRET")
    access_token = os.getenv("X_ACCESS_TOKEN")
    access_token_secret = os.getenv("X_ACCESS_TOKEN_SECRET")
    bearer_token = os.getenv("X_BEARER_TOKEN")

    # Create a client
    client = tweepy.Client(
        consumer_key=consumer_key,
        consumer_secret=consumer_secret,
        access_token=access_token,
        access_token_secret=access_token_secret,
        bearer_token=bearer_token
    )

    return client

def post_to_x(post):
    """
    Posts a tweet (X post) with the given text.

    Args:
        consumer_key (str): Your API key / consumer key.
        consumer_secret (str): Your API secret.
        access_token (str): Your access token.
        access_token_secret (str): Your access token secret.
        text (str): The text of the tweet.
    """

    client = get_x_client()
    # Create the tweet
    response = client.create_tweet(text=post)

    return response


def post_all_unposted(dataset_path="/workspace/RL_AI/data/dataset.jsonl"):
    """
    Posts all unposted tweets from dataset.jsonl and updates the dataset with tweet IDs.
    
    Args:
        dataset_path (str): Path to the dataset.jsonl file.
    
    Returns:
        dict: Summary with counts of posted and failed tweets.
    """
    # Read all entries
    entries = []
    with open(dataset_path, 'r') as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    
    posted_count = 0
    failed_count = 0
    
    # Post unposted entries
    for entry in entries:
        if not entry.get("posted", False):
            try:
                response = post_to_x(entry["post"])
                entry["tweet_id"] = response.data["id"]
                entry["posted"] = True
                posted_count += 1
                print(f"Posted: {entry['post'][:50]}... (ID: {entry['tweet_id']})")
            except Exception as e:
                failed_count += 1
                print(f"Failed to post: {entry['post'][:50]}... Error: {e}")
    
    # Write updated entries back
    with open(dataset_path, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    
    return {"posted": posted_count, "failed": failed_count, "total": len(entries)}


def get_x_metrics(tweet_id):
    """
    Retrieves views, likes, replies, reposts, quotes for a tweet.
    Note: Views require elevated API access (organic or non_public metrics).
    """
    client = get_x_client()
    response = client.get_tweet(
        id=tweet_id,
        # tweet_fields=["public_metrics", "non_public_metrics", "organic_metrics"] free api access only allows public_metrics
        tweet_fields=["public_metrics"]
    )

    print(response)
    data = response.data
    
    public = data["public_metrics"]

    # Views depend on your plan / access level.
    views = 0
    if "organic_metrics" in data:
        views = data["organic_metrics"].get("impression_count")
    elif "non_public_metrics" in data:
        views = data["non_public_metrics"].get("impression_count")

    return {
        "views": views,
        "likes": public.get("like_count"),
        "replies": public.get("reply_count"),
        "reposts": public.get("retweet_count"),
        "quotes": public.get("quote_count"),
    }


def update_all_metrics(dataset_path="/workspace/RL_AI/data/dataset.jsonl"):
    """
    Updates metrics for all posted tweets in dataset.jsonl.
    
    Args:
        dataset_path (str): Path to the dataset.jsonl file.
    
    Returns:
        dict: Summary with counts of updated and failed tweets.
    """
    # Read all entries
    entries = []
    with open(dataset_path, 'r') as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    
    updated_count = 0
    failed_count = 0
    
    # Update metrics for posted entries
    for entry in entries:
        if entry.get("posted", False) and entry.get("tweet_id") != "0":
            try:
                metrics = get_x_metrics(entry["tweet_id"])
                entry["views"] = metrics["views"]
                entry["likes"] = metrics["likes"]
                entry["reposts"] = metrics["reposts"]
                updated_count += 1
                print(f"Updated metrics for tweet {entry['tweet_id']}: {metrics}")
            except Exception as e:
                failed_count += 1
                print(f"Failed to get metrics for tweet {entry['tweet_id']}: {e}")
    
    # Write updated entries back
    with open(dataset_path, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    
    return {"updated": updated_count, "failed": failed_count, "total": len(entries)}