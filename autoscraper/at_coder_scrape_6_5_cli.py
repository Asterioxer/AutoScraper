import os
import datetime
import time
import requests
import pandas as pd
import typer
from dotenv import load_dotenv
from autoscraper.utils.logger import info, success, error
import json
from playwright.sync_api import sync_playwright
import cohere
import numpy as np
from sklearn.cluster import KMeans

load_dotenv()

app = typer.Typer(help="Phase 6.5: Problem Scraper + AI Teaching Insights")

# Constants
BASE_URL = "https://atcoder.jp/contests/"
PROBLEMSET_URL = "https://kenkoooo.com/atcoder/resources/problems.json"  # AtCoder problem list API
PROBLEM_PAGE = "https://atcoder.jp/contests/{}/tasks/{}"

# Initialize Cohere client
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise RuntimeError("COHERE_API_KEY not found in environment variables!")
co = cohere.Client(COHERE_API_KEY)


# ---------------------------------------------
# Fetch problem statements with Playwright
# ---------------------------------------------
def fetch_problem_html(url, retries=3, backoff=1):
    for attempt in range(1, retries + 1):
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.set_extra_http_headers({
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
                })
                info(f"Fetching problem page: {url} (attempt {attempt})")
                page.goto(url, timeout=15000)
                page.wait_for_selector("span.lang-en", timeout=7000)
                html = page.inner_html("span.lang-en")
                browser.close()
                return html
        except Exception as e:
            error(f"Error fetching {url}: {e}")
            if attempt < retries:
                wait = backoff * (2 ** (attempt - 1))
                info(f"Retrying in {wait} seconds...")
                time.sleep(wait)
            else:
                return None


# ---------------------------------------------
# AI: cluster and teaching transformation
# ---------------------------------------------
def cluster_problems_with_cohere(texts, k=5):
    if len(texts) < k:
        k = len(texts)
    info("Generating embeddings for clustering…")
    embeddings = co.embed(texts=texts, model="large").embeddings
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    return clusters


def enhance_problem_with_ai(statement_html):
    """
    Transform problem into a teaching resource with examples.
    """
    prompt = (
        "You are a teaching assistant. Simplify and explain the following programming problem for a beginner.\n"
        "1. Start with a simple summary of what is required.\n"
        "2. Break it into clear steps.\n"
        "3. Provide 2-3 clear sample input/output examples (invent if needed) with explanations.\n"
        "4. Make sure it's so clear that the reader does not need to re-check the original problem.\n\n"
        f"Problem:\n{statement_html}"
    )
    try:
        response = co.chat(
            model="command-r-plus",
            message=prompt,
            temperature=0.4
        )
        return response.text.strip()
    except Exception as e:
        error(f"Teaching transformation failed: {e}")
        return ""


# ---------------------------------------------
# Main pipeline
# ---------------------------------------------
@app.command()
def run_pipeline(max_problems: int = 5, clusters: int = 3):
    info(f"[PHASE 6.5] Starting AtCoder scrape + AI teaching transform for {max_problems} problems…")

    # Step 1: Fetch metadata
    resp = requests.get(PROBLEMSET_URL, timeout=15)
    resp.raise_for_status()
    all_problems = resp.json()
    problems = all_problems[:max_problems]

    # Step 2: Scrape statements
    problem_data = []
    for p in problems:
        contest_id = p["contest_id"]
        task_id = p["id"]
        name = p["title"]
        url = PROBLEM_PAGE.format(contest_id, task_id)
        html = fetch_problem_html(url)
        if html is None:
            html = ""
        problem_data.append({
            "contest_id": contest_id,
            "task_id": task_id,
            "name": name,
            "url": url,
            "statement": html
        })
        time.sleep(1)

    # Save raw
    folder = "phase65_runs"
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_csv = os.path.join(folder, f"raw_{timestamp}.csv")
    pd.DataFrame(problem_data).to_csv(raw_csv, index=False)
    success(f"Saved raw problems to {raw_csv}")

    # Step 3: Cluster
    statements_list = [p["statement"] or "empty" for p in problem_data]
    cluster_labels = cluster_problems_with_cohere(statements_list, k=clusters)
    for i, label in enumerate(cluster_labels):
        problem_data[i]["cluster"] = int(label)

    cluster_csv = os.path.join(folder, f"clustered_{timestamp}.csv")
    pd.DataFrame(problem_data).to_csv(cluster_csv, index=False)
    success(f"Saved clustered problems to {cluster_csv}")

    # Step 4: Transform each with teaching AI
    for p in problem_data:
        p["teaching_version"] = enhance_problem_with_ai(p["statement"])

    # Save teaching version
    final_csv = os.path.join(folder, f"teaching_{timestamp}.csv")
    final_json = os.path.join(folder, f"teaching_{timestamp}.json")
    pd.DataFrame(problem_data).to_csv(final_csv, index=False)
    pd.DataFrame(problem_data).to_json(final_json, orient="records", indent=2)
    success(f"[PHASE 6.5] Teaching-enhanced problems saved to {final_csv} and {final_json}")
    success("🚀🔥 Phase 6.5 pipeline completed successfully!")


if __name__ == "__main__":
    app()
