import requests
from datetime import datetime

def fetch_live_reddit_posts(subreddit="all", limit=30):
    """
    Fetches the latest posts from a given subreddit without authentication
    using the public JSON endpoint.
    """
    url = f"https://www.reddit.com/r/{subreddit}/new.json?limit={limit}"
    # Adding a custom user-agent to avoid 429 Too Many Requests
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) dsbda-bot/1.0"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        posts = []
        for child in data.get("data", {}).get("children", []):
            post_data = child.get("data", {})
            title = post_data.get("title", "")
            text = post_data.get("selftext", "")
            # Combine title and selftext for analysis
            full_text = f"{title} {text}".strip()
            
            # Skip empty posts
            if not full_text:
                continue
                
            posts.append({
                "id": post_data.get("id"),
                "text": full_text,
                "author": post_data.get("author"),
                "created_utc": post_data.get("created_utc", 0),
                "timestamp": datetime.fromtimestamp(post_data.get("created_utc", 0)),
                "url": f"https://www.reddit.com{post_data.get('permalink')}",
                "subreddit": post_data.get("subreddit")
            })
        
        # Sort by timestamp, newest first
        posts = sorted(posts, key=lambda x: x["created_utc"], reverse=True)
        return posts
        
    except Exception as e:
        print(f"Error fetching live data: {e}")
        return []

def search_live_reddit_posts(query: str, limit=30):
    """
    Searches Reddit for the latest posts containing the given query.
    """
    import urllib.parse
    encoded_query = urllib.parse.quote(query)
    url = f"https://www.reddit.com/search.json?q={encoded_query}&sort=new&limit={limit}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) dsbda-bot/1.0"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        posts = []
        for child in data.get("data", {}).get("children", []):
            post_data = child.get("data", {})
            title = post_data.get("title", "")
            text = post_data.get("selftext", "")
            full_text = f"{title} {text}".strip()
            
            if not full_text:
                continue
                
            posts.append({
                "id": post_data.get("id"),
                "text": full_text,
                "author": post_data.get("author"),
                "created_utc": post_data.get("created_utc", 0),
                "timestamp": datetime.fromtimestamp(post_data.get("created_utc", 0)),
                "url": f"https://www.reddit.com{post_data.get('permalink')}",
                "subreddit": post_data.get("subreddit")
            })
        
        posts = sorted(posts, key=lambda x: x["created_utc"], reverse=True)
        return posts
        
    except Exception as e:
        print(f"Error searching live data for {query}: {e}")
        return []
