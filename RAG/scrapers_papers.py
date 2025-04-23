# Re-import necessary packages after kernel reset
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
from Bio import Entrez
from serpapi import GoogleSearch
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set your email for PubMed Entrez from environment variables
Entrez.email = os.getenv("PUBMED_EMAIL", "pes1202203223@pesu.pes.edu")

# Set your SerpAPI key from environment variables
api_key = os.getenv("SERPAPI_KEY")

# Set up base directories
os.makedirs("RAG/kb_docs/arxiv", exist_ok=True)
os.makedirs("RAG/kb_docs/pubmed", exist_ok=True)

# Keywords to search
keywords = ["music emotion therapy", "valence arousal dominance", "music affective computing", "emotion based music recommendation"]

# ---- ARXIV SCRAPER ----
def scrape_arxiv(keyword, max_results=50):
    print(f"[arXiv] Scraping for keyword: {keyword}")
    query = quote(keyword)
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"
    resp = requests.get(url)
    soup = BeautifulSoup(resp.content, "xml")
    entries = soup.find_all("entry")
    
    for idx, entry in enumerate(entries):
        title = entry.title.text.strip().replace('\n', ' ')
        abstract = entry.summary.text.strip().replace('\n', ' ')
        link = entry.id.text.strip()
        content = f"Title: {title}\n\nAbstract: {abstract}\n\nLink: {link}"
        
        with open(f"RAG/kb_docs/arxiv/{keyword[:50].replace(' ', '_')}_{idx}.txt", "w") as f:
            f.write(content)

# ---- PUBMED SCRAPER ----
def scrape_pubmed(keyword, max_results=50):
    print(f"[PubMed] Scraping for keyword: {keyword}")
    handle = Entrez.esearch(db="pubmed", term=keyword, retmax=max_results)
    record = Entrez.read(handle)
    id_list = record["IdList"]

    for idx, pubmed_id in enumerate(id_list):
        fetch_handle = Entrez.efetch(db="pubmed", id=pubmed_id, rettype="abstract", retmode="text")
        text = fetch_handle.read()
        with open(f"RAG/kb_docs/pubmed/{keyword[:50].replace(' ', '_')}_{idx}.txt", "w") as f:
            f.write(text)

# ---- MASTER SCRAPE RUNNER ----
def run_scrapers():
    for kw in keywords:
        scrape_arxiv(kw)
        scrape_pubmed(kw)

# Uncomment the following line when you're ready to run scraping:
run_scrapers()
