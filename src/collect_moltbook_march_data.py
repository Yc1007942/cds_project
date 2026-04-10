import requests
import json
import time
from datetime import datetime

BASE_URL = "https://www.moltbook.com/api/v1"
SUBMOLTS = ["philosophy", "todayilearned", "technology"]
OUTPUT_FILE = "../data/moltbook_3month_2026_data_all2.json"

REQUEST_DELAY = 0.5          # seconds between requests (increased from 0.1)
MAX_RETRIES = 3               # number of retries for failed requests

START_DATE = datetime(2026, 1, 1)
END_DATE = datetime(2026, 3, 31, 23, 59, 59)

def is_in_march_2026(date_str):
    try:
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        dt = dt.replace(tzinfo=None)
        return START_DATE <= dt <= END_DATE
    except Exception:
        return False

def is_before_march_2026(date_str):
    try:
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        dt = dt.replace(tzinfo=None)
        return dt < START_DATE
    except Exception:
        return False

def fetch_with_retries(url, retries=MAX_RETRIES):
    """Generic GET with exponential backoff for 5xx errors."""
    for attempt in range(retries):
        try:
            response = requests.get(url)
            # Retry on server errors (5xx)
            if 500 <= response.status_code < 600:
                raise Exception(f"Server error {response.status_code}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if attempt == retries - 1:
                print(f"  Failed after {retries} attempts: {e}")
                return {}
            wait = 2 ** attempt  # 1, 2, 4 seconds
            print(f"  Retry {attempt+1}/{retries} in {wait}s...")
            time.sleep(wait)
    return {}

def fetch_posts_page(submolt_name, cursor=None):
    url = f"{BASE_URL}/posts?submolt={submolt_name}"
    if cursor:
        url += f"&cursor={cursor}"
    return fetch_with_retries(url)

def fetch_comments(post_id):
    url = f"{BASE_URL}/posts/{post_id}/comments"
    return fetch_with_retries(url)

def save_all_data(data):
    """Write the entire data list to the output file."""
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)

def collect_march_data():
    all_data = []
    # Load existing data if file exists (optional, to resume)
    try:
        with open(OUTPUT_FILE, "r") as f:
            all_data = json.load(f)
        print(f"Loaded {len(all_data)} existing records from {OUTPUT_FILE}")
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    for submolt in SUBMOLTS:
        print(f"\nCollecting March 2026 data from m/{submolt}...")
        cursor = None
        stop_submolt = False

        while not stop_submolt:
            page_data = fetch_posts_page(submolt, cursor)
            posts = page_data.get("posts", [])
            if not posts:
                break

            for post in posts:
                created_at = post.get("created_at")
                if is_before_march_2026(created_at):
                    print(f"  Reached data before March 2026. Stopping m/{submolt}.")
                    stop_submolt = True
                    break

                if is_in_march_2026(created_at):
                    post_id = post.get("id")
                    print(f"  Fetching comments for: {post.get('title')[:50]}...")
                    comments = fetch_comments(post_id)
                    time.sleep(REQUEST_DELAY)   # delay after fetching comments

                    march_comments = [
                        {
                            "id": c.get("id"),
                            "created_at": c.get("created_at"),
                            "score": c.get("score"),
                            "content": c.get("content"),
                            "author": c.get("author", {}).get("name")
                        }
                        for c in comments.get("comments", [])  # fetch_comments returns dict with 'comments'
                        if is_in_march_2026(c.get("created_at"))
                    ]

                    structured_post = {
                        "created_at": created_at,
                        "id": post_id,
                        "comment_count": post.get("comment_count"),
                        "score": post.get("score"),
                        "forum": submolt,
                        "post": {
                            "title": post.get("title"),
                            "content": post.get("content")
                        },
                        "comments": march_comments
                    }
                    all_data.append(structured_post)
                    # Save after each post – fine‑grained persistence
                    save_all_data(all_data)
                    print(f"    Saved post {post_id} (total: {len(all_data)})")
                    time.sleep(REQUEST_DELAY)   # delay between posts
                else:
                    # Post is after March? Should not happen if ordering is newest first,
                    # but just in case we skip.
                    pass

            # Move to next page
            cursor = page_data.get("next_cursor")
            if not cursor or not page_data.get("has_more"):
                break

    print(f"\nMarch 2026 data collection complete. Total posts: {len(all_data)}. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    collect_march_data()