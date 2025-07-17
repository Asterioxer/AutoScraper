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
app = typer.Typer(help="AtCoder/Codeforces Scraper + AI Insights + Split by Tag/Difficulty + Starter Templates")

# ========================= CONFIG =========================
# Example using AtCoder API for reliability
API_URL = "https://kenkoooo.com/atcoder/resources/problems.json"
BASE_PROBLEM_URL = "https://atcoder.jp/contests"
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise RuntimeError("COHERE_API_KEY not found in environment variables!")
co = cohere.Client(COHERE_API_KEY)
# ==========================================================

def fetch_problems(max_problems):
    info(f"[Phase 6.6] Fetching top {max_problems} problems from AtCoder API…")
    resp = requests.get(API_URL, timeout=15)
    resp.raise_for_status()
    problems = resp.json()[:max_problems]
    # Add url field
    for p in problems:
        contest_id = p.get("contest_id")
        task_id = p.get("id")
        p["url"] = f"{BASE_PROBLEM_URL}/{contest_id}/tasks/{task_id}"
    return problems

def fetch_problem_statement_playwright(problem):
    contest_id = problem.get("contest_id")
    task_id = problem.get("id")
    url = f"{BASE_PROBLEM_URL}/{contest_id}/tasks/{task_id}"

    info(f"[FETCH] Opening {url}")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)  # set False if you want to see it
        page = browser.new_page()
        page.set_default_timeout(600)  # extend timeout
        try:
            page.goto(url, timeout=600)
            # Wait for the main problem container
            page.wait_for_selector("div.part", timeout=600)
            html = page.content()
        except Exception as e:
            error(f"[FETCH FAIL] {task_id}: {e}")
            html = ""
        finally:
            browser.close()
        return html


def cluster_problems_with_cohere(texts, k=5):
    if len(texts) < k:
        k = len(texts)
    info("Clustering with Cohere embeddings...")
    embeddings = co.embed(texts=texts, model="embed-english-light-v3.0").embeddings
    kmeans = KMeans(n_clusters=k, random_state=42)
    return kmeans.fit_predict(embeddings)

def generate_teaching_version(text):
    prompt = f"""Simplify the following problem for a beginner:
{text}

Include:
1. A clear, beginner-friendly explanation.
2. Key points to focus on.
3. A sample input/output with an explanation.
"""
    resp = co.chat(model="command-r-plus", message=prompt)
    return resp.text.strip()

def generate_starter_code(teaching_version):
    prompt = f"""Given the following problem details:

{teaching_version}

Generate a Python starter code template with:
- Reading input from stdin
- Writing output to stdout
- A function skeleton with TODOs
"""
    resp = co.chat(model="command-r-plus", message=prompt)
    return resp.text.strip()

@app.command()
def run_pipeline(max_problems: int = 10, clusters: int = 3):
    info(f"[Phase 6.6] Starting pipeline for {max_problems} problems")

    problems = fetch_problems(max_problems)
    statements = []
    for p in problems:
        try:
            html = fetch_problem_statement_playwright(p)
            statements.append(html)
        except Exception as e:
            error(f"Failed fetching {p.get('id')}: {e}")
            statements.append("")

    # Attach statements
    for p, s in zip(problems, statements):
        p["statement"] = s

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = f"phase6_6_runs_{timestamp}"
    os.makedirs(folder, exist_ok=True)

    # Cluster
    labels = cluster_problems_with_cohere([p["statement"] for p in problems], clusters)
    for i, lbl in enumerate(labels):
        problems[i]["cluster"] = int(lbl)

    # Generate teaching versions and starter codes
    starter_folder = os.path.join(folder, "starter_codes")
    os.makedirs(starter_folder, exist_ok=True)
    for p in problems:
        tv = generate_teaching_version(p["statement"])
        p["teaching_version"] = tv
        starter = generate_starter_code(tv)
        p["starter_code"] = starter
        with open(os.path.join(starter_folder, f"{p['id']}.py"), "w", encoding="utf-8") as f:
            f.write(starter)

    # Save grouped by tags/difficulty
    by_tag = {}
    for p in problems:
        tags = p.get("tags", [])
        if not tags:
            tags = ["untagged"]
        for tag in tags:
            by_tag.setdefault(tag, []).append(p)
    for tag, group in by_tag.items():
        safe_tag = tag.replace("/", "_")
        pd.DataFrame(group).to_csv(os.path.join(folder, f"group_tag_{safe_tag}.csv"), index=False)

    # Save everything
    pd.DataFrame(problems).to_csv(os.path.join(folder, "all_problems.csv"), index=False)
    with open(os.path.join(folder, "all_problems.json"), "w", encoding="utf-8") as f:
        json.dump(problems, f, indent=2, ensure_ascii=False)

    success(f"[Phase 6.6] All data saved in {folder}")
    success(f"[Phase 6.6] Starter templates saved in {starter_folder}")
    success("[Phase 6.6] Pipeline complete 🚀🔥")

if __name__ == "__main__":
    app()
