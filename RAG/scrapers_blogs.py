import os
import requests
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
import time
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup directories
os.makedirs("RAG/kb_docs/blogs", exist_ok=True)

# Keywords to search
keywords = [
    "music emotion therapy", 
    "valence arousal dominance", 
    "music affective computing", 
    "emotion based music recommendation", 
    "musical counselling"
]

# SerpAPI Key from environment variables
api_key = os.getenv("SERPAPI_KEY")

# Extract full text (cleaned) from blog/article
def extract_full_text_from_blog(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.content, "html.parser")

        # Remove common ad/script/style tags
        for tag in soup(['script', 'style', 'noscript', 'header', 'footer', 'svg', 'img', 'iframe', 'nav', 'form', 'aside']):
            tag.decompose()

        # Get visible text
        text = soup.get_text(separator="\n")

        # Remove excessive whitespace and non-content lines
        lines = [line.strip() for line in text.splitlines()]
        clean_text = "\n".join([line for line in lines if len(line) > 30])  # filter very short lines

        title = soup.title.string.strip() if soup.title else "No Title"
        return title, clean_text
    
    except Exception as e:
        return "No Title", f"Failed to extract from {url} due to {str(e)}"

# Blog scraping using SerpAPI Google Search
def scrape_blogs_for_keyword(keyword, max_results=10):
    print(f"[Blogs] Searching for: {keyword}")
    params = {
        "engine": "google",
        "q": keyword + " site:medium.com OR site:psychologytoday.com OR site:researchgate.net OR site:blogs.scientificamerican.com",
        "api_key": api_key,
        "num": max_results
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    for idx, result in enumerate(results.get("organic_results", [])):
        link = result.get("link")
        if not link:
            continue

        title, full_text = extract_full_text_from_blog(link)

        content = f"""Title: {title}
Link: {link}

--- Full Text Start ---

{full_text}

--- End ---
"""

        filename = f"RAG/kb_docs/blogs/{keyword[:50].replace(' ', '_')}_{idx}.txt"
        with open(filename, "w", encoding='utf-8') as f:
            f.write(content)

        time.sleep(1.5)  # Respect API rate limits

# Run scraper on all keywords
def run_blog_scraper():
    for kw in keywords:
        scrape_blogs_for_keyword(kw)

# Run it!
run_blog_scraper()
