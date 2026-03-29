import asyncio
import aiohttp
import json
from datetime import datetime
import sys

BASE_URL = "https://www.moltbook.com/api/v1"
SUBMOLTS = ["philosophy", "todayilearned", "technology"]
OUTPUT_FILE = "../data/moltbook_3month_2026_data_all_4.json"

# Limits how many requests happen at the exact same time
CONCURRENCY_LIMIT = 10 

START_DATE = datetime(2026, 1, 1)
END_DATE = datetime(2026, 3, 31, 23, 59, 59)

# Global variables for safe file writing
all_data = []
file_write_lock = asyncio.Lock()

def is_in_march_2026(date_str):
    dt = datetime.fromisoformat(date_str.replace('Z', '+00:00')).replace(tzinfo=None)
    return START_DATE <= dt <= END_DATE

def is_before_march_2026(date_str):
    dt = datetime.fromisoformat(date_str.replace('Z', '+00:00')).replace(tzinfo=None)
    return dt < START_DATE

async def fetch_with_retries(session, url, retries=3):
    """Generic async GET with backoff for failed requests."""
    for attempt in range(retries):
        async with session.get(url) as response:
            if response.status == 200:
                return await response.json()
            
            # Pause exponentially before retrying (1s, 2s, 4s...)
            await asyncio.sleep(2 ** attempt)
    return {}

async def save_data_safe():
    """Safely writes the global list to the JSON file using an async lock."""
    async with file_write_lock:
        with open(OUTPUT_FILE, "w") as f:
            json.dump(all_data, f, indent=2)

async def process_post(session, post, submolt, semaphore):
    """Fetches comments for a post, formats the data, and triggers a safe save."""
    async with semaphore:
        post_id = post.get("id")
        url = f"{BASE_URL}/posts/{post_id}/comments"
        
        print(f"  Fetching comments for: {post.get('title')[:50]}...")
        comments_data = await fetch_with_retries(session, url)

        # Filter and format the comments
        march_comments = [
            {
                "id": c.get("id"),
                "created_at": c.get("created_at"),
                "score": c.get("score"),
                "content": c.get("content"),
                "author": c.get("author", {}).get("name") if c.get("author") else None
            }
            for c in comments_data.get("comments", [])
            if is_in_march_2026(c.get("created_at"))
        ]

        # Format the post
        structured_post = {
            "created_at": post.get("created_at"),
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

        # Append to memory and save to disk
        all_data.append(structured_post)
        await save_data_safe()
        print(f"    Saved post {post_id} (total: {len(all_data)})")

async def scrape_submolt(session, submolt, semaphore):
    """Paginates through a submolt and triggers concurrent post processing."""
    print(f"\nCollecting March 2026 data from m/{submolt}...")
    cursor = None
    stop_submolt = False
    tasks = []

    while not stop_submolt:
        url = f"{BASE_URL}/posts?submolt={submolt}"
        if cursor:
            url += f"&cursor={cursor}"

        page_data = await fetch_with_retries(session, url)
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
                # Create an asynchronous task for this post so it runs in the background
                task = asyncio.create_task(process_post(session, post, submolt, semaphore))
                tasks.append(task)

        cursor = page_data.get("next_cursor")
        if not cursor or not page_data.get("has_more"):
            break

    # Wait for all the comment-fetching background tasks to finish for this submolt
    if tasks:
        await asyncio.gather(*tasks)

async def main():
    # Restrict to 10 simultaneous network requests
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    # Open an async HTTP session that stays alive for all requests
    async with aiohttp.ClientSession() as session:
        # Run the submolt scrapers concurrently
        submolt_tasks = [scrape_submolt(session, submolt, semaphore) for submolt in SUBMOLTS]
        await asyncio.gather(*submolt_tasks)
        
    print(f"\nJanuary-March 2026 data collection complete. Total posts: {len(all_data)}. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(main())