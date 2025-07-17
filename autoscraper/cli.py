import typer
import json
import csv
from autoscraper.core.scraper import scrape_with_pagination
from autoscraper.core.classifier import SimpleClassifier
from autoscraper.core.eda import run_eda
from autoscraper.utils.logger import info, success, error
from autoscraper.core.ai_insights import run_ai_insights
from autoscraper.core.gpt_cluster_describer import describe_clusters

app = typer.Typer()

@app.command()
def fetch(url: str, selector: str):
    """Fetch data directly by passing URL and single CSS selector."""
    data = scrape_with_pagination(url, {"data": selector}, max_pages=1)
    for item in data[:10]:
        print("→", item.get("data"))
    success(f"Scraped {len(data)} items.")

@app.command()
def run_config(
    config_path: str,
    csv_output: bool = typer.Option(False, "--csv", help="Export as CSV instead of JSON"),
    max_pages: int = typer.Option(3, "--max-pages", help="Max pages to scrape"),
    output_path: str = typer.Option(None, "--output", help="Output file path"),
    retries: int = typer.Option(3, "--retries", help="Number of retries for HTTP requests"),
    timeout: int = typer.Option(10, "--timeout", help="Timeout (seconds) for HTTP requests"),
):
    """Run scraper based on a JSON config file."""
    info(f"Loading config from {config_path}")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        error(f"Failed to load config: {e}")
        raise typer.Exit(code=1)

    url = config.get("url")
    selectors = config.get("selectors", {})
    pagination_selector = config.get("pagination")

    if not url or not selectors:
        error("Config must include 'url' and 'selectors'")
        raise typer.Exit(code=1)

    info(f"Scraping from {url} with pagination up to {max_pages} pages...")
    data = scrape_with_pagination(
        url,
        selectors,
        pagination_selector,
        max_pages=max_pages,
        retries=retries,
        timeout=timeout,
    )

    # --- Phase 4: Classification step ---
    classifier = SimpleClassifier()
    for row in data:
        row["predicted_categories"] = classifier.classify(row)

    # preview
    for r in data[:5]:
        print("→", r)
    success(f"Scraped {len(data)} rows.")

    # set default output if not provided
    if not output_path:
        output_path = "output.csv" if csv_output else "output.json"

    if csv_output:
        fieldnames = list(selectors.keys()) + ["predicted_categories"]
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow(row)
        success(f"Saved CSV to {output_path}")
    else:
        with open(output_path, "w", encoding="utf-8") as out:
            json.dump(data, out, indent=2, ensure_ascii=False)
        success(f"Saved JSON to {output_path}")

@app.command()
def eda(
    input_csv: str = typer.Option("output.csv", help="Input CSV from scraper"),
    cleaned_csv: str = typer.Option("output_cleaned.csv", help="Path to save cleaned CSV"),
    summary_json: str = typer.Option("insights.json", help="Path to save insights JSON"),
):
    """Run EDA & cleaning on a scraped CSV."""
    run_eda(input_csv, cleaned_csv, summary_json)
    success("EDA completed successfully!")

@app.command()
def ai_insights(
    input_csv: str = typer.Option("output_cleaned.csv", help="Cleaned CSV from EDA"),
    output_json: str = typer.Option("ai_insights.json", help="AI Insights JSON output"),
    clusters: int = typer.Option(5, help="Number of clusters to group into")
):
    """Generate AI-driven clustering insights from scraped data."""
    run_ai_insights(input_csv, output_json, clusters)
    success("AI insights generation completed!")

@app.command()
def gpt_describe_clusters(
    input_csv: str = typer.Option("output_ai_tagged.csv", help="CSV with ai_cluster labels"),
    output_json: str = typer.Option("cluster_descriptions.json", help="Where to save cluster summaries"),
    top_n: int = typer.Option(5, help="Top N texts per cluster to use for summarization")
):
    """
    Use Cohere to generate natural language descriptions for each cluster (Phase 5).
    """
    describe_clusters(input_csv, output_json, top_n)
    success("Cluster descriptions generated successfully!")
if __name__ == "__main__":
    app()
