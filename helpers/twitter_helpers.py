import os, requests

API_BASE = "https://api.x.com/2"

def _auth_headers():
    token = os.environ.get("X_BEARER_TOKEN")
    if not token:
        raise RuntimeError("Missing X_BEARER_TOKEN in environment.")
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

def post_to_x(text: str) -> str:
    """
    Create a post (tweet) via X API v2.
    Requires a user-context OAuth 2.0 Bearer token in X_BEARER_TOKEN.
    """
    url = f"{API_BASE}/tweets"
    body = {"text": text}
    resp = requests.post(url, headers=_auth_headers(), json=body, timeout=30)
    if resp.status_code >= 400:
        raise RuntimeError(f"POST {url} failed: {resp.status_code} {resp.text}")
    data = resp.json()
    return str(data["data"]["id"])

def fetch_metrics_x(tweet_id: str) -> dict:
    """
    Read a post and return public metrics.
    Returns dict with keys: 'quotes' (impressions if available), 'likes', 'reposts'.
    """
    url = f"{API_BASE}/tweets/{tweet_id}"
    params = {"tweet.fields": "public_metrics,created_at"}
    resp = requests.get(url, headers=_auth_headers(), params=params, timeout=30)
    if resp.status_code >= 400:
        raise RuntimeError(f"GET {url} failed: {resp.status_code} {resp.text}")
    data = resp.json()
    m = data.get("data", {}).get("public_metrics", {}) or {}
    # Maintain existing contract used by helper_functions.update_post_metrics:
    # - 'quotes' was previously treated as 'views'; map to impression_count if present.
    impressions = m.get("impression_count", 0)
    likes = m.get("like_count", 0)
    reposts = m.get("retweet_count", 0)
    return {"quotes": impressions, "likes": likes, "reposts": reposts}