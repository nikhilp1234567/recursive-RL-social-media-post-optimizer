import os
import tweepy

def get_x_client():
    return tweepy.Client(
        bearer_token=os.environ["X_BEARER_TOKEN"],
        consumer_key=os.environ["X_API_KEY"],
        consumer_secret=os.environ["X_API_SECRET"],
        access_token=os.environ["X_ACCESS_TOKEN"],
        access_token_secret=os.environ["X_ACCESS_SECRET"],
    )

def post_to_x(text: str) -> str:
    client = get_x_client()
    resp = client.create_tweet(text=text)
    return str(resp.data["id"])

def fetch_metrics_x(tweet_id: str) -> dict:
    client = get_x_client()
    resp = client.get_tweet(id=tweet_id, tweet_fields=["public_metrics"])
    m = resp.data.public_metrics
    return {"quotes": m.get("impression_count", 0), "likes": m.get("like_count", 0), "reposts": m.get("retweet_count", 0)}