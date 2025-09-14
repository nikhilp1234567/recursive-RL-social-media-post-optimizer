import os
import tweepy

def get_x_client():
    """
    Create and return a Tweepy Client for interacting with the X (Twitter) API.

    Returns:
        tweepy.Client: An authenticated Tweepy client using credentials from environment variables.
    """
    return tweepy.Client(
        bearer_token=os.environ["X_BEARER_TOKEN"],
        consumer_key=os.environ["X_API_KEY"],
        consumer_secret=os.environ["X_API_SECRET"],
        access_token=os.environ["X_ACCESS_TOKEN"],
        access_token_secret=os.environ["X_ACCESS_SECRET"],
    )

def post_to_x(text: str) -> str:
    """
    Post a tweet to X (Twitter) using the provided text.

    Args:
        text (str): The content of the tweet to post.

    Returns:
        str: The ID of the posted tweet.
    """
    client = get_x_client()
    resp = client.create_tweet(text=text)
    return str(resp.data["id"])

def fetch_metrics_x(tweet_id: str) -> dict:
    """
    Fetch public metrics for a given tweet from X (Twitter).

    Args:
        tweet_id (str): The ID of the tweet to fetch metrics for.

    Returns:
        dict: A dictionary containing the number of quotes (impressions), likes, and reposts (retweets).
              Keys are "quotes", "likes", and "reposts".
    """
    client = get_x_client()
    resp = client.get_tweet(id=tweet_id, tweet_fields=["public_metrics"])
    m = resp.data.public_metrics
    return {"quotes": m.get("impression_count", 0), "likes": m.get("like_count", 0), "reposts": m.get("retweet_count", 0)}