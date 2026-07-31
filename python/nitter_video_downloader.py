"""
Nitter RSS Video Scraper.

Scrapes public Nitter instances for tweets containing video files matching
the search query, parses the video source URLs, and downloads them.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import urllib.request
import urllib.parse
from typing import List

# Ensure we have feedparser
try:
    import feedparser
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "feedparser"], check=True)
    import feedparser

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("nitter_scraper")

# List of currently active public Nitter instances
NITTER_INSTANCES = [
    "nitter.poast.org",
    "nitter.privacydev.net",
    "nitter.cz",
    "nitter.no-logs.wtf",
    "nitter.net"
]


def extract_video_urls_from_description(html_content: str, instance: str) -> List[str]:
    """Parses HTML description from Nitter feed entry to find direct video MP4 links."""
    # Nitter typically embeds videos using <video> tags or absolute paths like /video.mp4
    urls = []
    # Search for source tags or direct video links
    matches = re.findall(r'src=["\'](https://[^"\']+\.mp4|/[^"\']+\.mp4)["\']', html_content)
    for m in matches:
        if m.startswith("/"):
            urls.append(f"https://{instance}{m}")
        else:
            urls.append(m)
            
    # Fallback to general video links in anchors
    matches_href = re.findall(r'href=["\'](https://[^"\']+\.mp4|/[^"\']+\.mp4)["\']', html_content)
    for m in matches_href:
        if m.startswith("/"):
            urls.append(f"https://{instance}{m}")
        else:
            urls.append(m)
            
    return list(set(urls))


def fetch_nitter_videos(query: str, download_dir: str, limit: int = 5) -> List[str]:
    """Searches for query on Nitter instances, extracts video URLs, and downloads them."""
    os.makedirs(download_dir, exist_ok=True)
    downloaded_paths = []
    
    encoded_query = urllib.parse.quote(query)
    
    for instance in NITTER_INSTANCES:
        # Search RSS feed url format
        rss_url = f"https://{instance}/search/rss?q={encoded_query}"
        logger.info(f"Attempting to fetch RSS from: {rss_url}")
        
        try:
            # Add User-Agent header to bypass simple blocks
            req = urllib.request.Request(
                rss_url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )
            with urllib.request.urlopen(req, timeout=15) as response:
                feed_data = response.read()
                
            feed = feedparser.parse(feed_data)
            
            if not feed.entries:
                logger.warning(f"No entries found on {instance}")
                continue
                
            logger.info(f"Found {len(feed.entries)} entries on {instance}. Scraping video links...")
            
            for entry in feed.entries:
                description = getattr(entry, 'description', '')
                video_urls = extract_video_urls_from_description(description, instance)
                
                for v_url in video_urls:
                    logger.info(f"Found video URL: {v_url}")
                    file_name = f"video_{len(downloaded_paths)+1}.mp4"
                    dest_path = os.path.join(download_dir, file_name)
                    
                    try:
                        logger.info(f"Downloading {v_url} to {dest_path}...")
                        v_req = urllib.request.Request(
                            v_url,
                            headers={'User-Agent': 'Mozilla/5.0'}
                        )
                        with urllib.request.urlopen(v_req, timeout=20) as v_resp, open(dest_path, "wb") as f_out:
                            f_out.write(v_resp.read())
                            
                        if os.path.getsize(dest_path) > 100000: # Ensure it is not an empty file
                            downloaded_paths.append(dest_path)
                            logger.info(f"Successfully downloaded: {dest_path}")
                        else:
                            os.remove(dest_path)
                            
                        if len(downloaded_paths) >= limit:
                            return downloaded_paths
                            
                    except Exception as dl_err:
                        logger.error(f"Failed to download video from {v_url}: {dl_err}")
                        
            if downloaded_paths:
                return downloaded_paths
                
        except Exception as e:
            logger.error(f"Failed to connect to Nitter instance {instance}: {e}")
            
    return downloaded_paths


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", "-q", default="swimming fails video", help="Search query")
    parser.add_argument("--output-dir", "-o", default="assets/videos", help="Download directory")
    args = parser.parse_args()
    
    paths = fetch_nitter_videos(args.query, args.output_dir)
    print(f"Downloaded {len(paths)} videos.")
