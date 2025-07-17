import os
import datetime
import pandas as pd
import typer
from dotenv import load_dotenv

# --- Autoscraper internal imports ---
from autoscraper.utils.logger import info, success, error
from autoscraper.core.scraper import scrape_with_pagination
from autoscraper.core.eda import run_eda
from autoscraper.core.enricher import semantic_enrich
from autoscraper.core.ai_insights import run_ai_insights
from autoscraper.core.gpt_cluster_describer import describe_clusters

# --- Load environment ---
load_dotenv()
import cohere
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise RuntimeError("COHERE_API_KEY not found in environment variables!")
co = cohere.Client(COHERE_API_KEY)

# --- Typer app ---
app = typer.Typer(help="Random URL scraping and AI enrichment CLI")

@app.command()
def randomurl(
    url: str = typer.Option(..., help="Base URL to scrape"),
    selector: str = typer.Option(..., help="CSS selector for data items"),
    pagination_selector: str = typer.Option(None, help="CSS selector for pagination link"),
    max_pages: int = typer.Option(3, help="Max pages to scrape"),
    sim_threshold: float = typer.Option(0.9, help="Similarity threshold for semantic enrichment"),
    clusters: int = typer.Option(5, help="Number of clusters for AI insights"),
    top_n: int = typer.Option(5, help="Top N examples per cluster for summaries"),
    model: str = typer.Option("command-xlarge", help="Cohere model to use for cluster description"),
):
    """
    Phase 6.2 Extended:
    Scrape -> Clean -> Enrich -> Cluster -> Cohere Summarize
    Each run outputs to its own timestamped files in 'randomurl_runs'.
    """
    try:
        # Prepare folder and timestamp
        folder = "randomurl_runs"
        os.makedirs(folder, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        info(f"[PHASE 6.2] Starting randomurl pipeline for {url}")

        # STEP 1: Scrape
        selectors = {"data": selector}
        scraped_data = scrape_with_pagination(
            base_url=url,
            selectors=selectors,
            pagination_selector=pagination_selector,
            max_pages=max_pages,
            retries=3,
            timeout=10
        )
        if not scraped_data:
            error("No data scraped. Check URL or selector.")
            raise typer.Exit(code=1)

        raw_csv = os.path.join(folder, f"randomurl_raw_{timestamp}.csv")
        pd.DataFrame(scraped_data).to_csv(raw_csv, index=False, encoding="utf-8")
        success(f"Raw scraped data saved to {raw_csv} (rows: {len(scraped_data)})")

        # STEP 2: EDA Cleaning
        cleaned_csv = os.path.join(folder, f"randomurl_cleaned_{timestamp}.csv")
        summary_json = os.path.join(folder, f"randomurl_summary_{timestamp}.json")
        run_eda(raw_csv, cleaned_csv, summary_json)
        success(f"Cleaned CSV saved to {cleaned_csv}")
        success(f"EDA summary JSON saved to {summary_json}")

        # STEP 3: Semantic Enrichment
        enriched_csv = os.path.join(folder, f"randomurl_enriched_{timestamp}.csv")
        semantic_enrich(cleaned_csv, enriched_csv, sim_threshold)
        success(f"Enriched CSV saved to {enriched_csv}")

        # STEP 4: AI Clustering
        ai_insights_json = os.path.join(folder, f"randomurl_ai_insights_{timestamp}.json")
        run_ai_insights(enriched_csv, ai_insights_json, clusters)

        # Move/rename the default output to a unique run-based file
        clustered_csv = os.path.join(folder, f"randomurl_clustered_{timestamp}.csv")
        if os.path.exists("output_ai_tagged.csv"):
            os.replace("output_ai_tagged.csv", clustered_csv)
            success(f"Clustered CSV saved to {clustered_csv}")
        else:
            error("Expected clustered file 'output_ai_tagged.csv' not found after clustering.")
            raise typer.Exit(code=1)

        success(f"AI insights JSON saved to {ai_insights_json}")

        # STEP 5: Cohere cluster summaries
        cluster_descriptions_json = os.path.join(folder, f"randomurl_cluster_descriptions_{timestamp}.json")
        describe_clusters(clustered_csv, cluster_descriptions_json, top_n=top_n, model=model)
        success(f"Cluster descriptions saved to {cluster_descriptions_json}")

        success("[PHASE 6.2] Randomurl full pipeline completed successfully! 🎯🚀")

    except Exception as e:
        error(f"Randomurl pipeline failed: {e}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
