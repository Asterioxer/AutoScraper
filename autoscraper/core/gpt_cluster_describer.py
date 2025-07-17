import pandas as pd
import json
from autoscraper.utils.logger import info, success, error
from collections import defaultdict
from dotenv import load_dotenv
import os
import cohere  # ✅ switched from openai to cohere

# Load .env from the project root
load_dotenv()

# Set the Cohere key
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise RuntimeError("COHERE_API_KEY not found in environment variables!")

# Initialize Cohere client
co = cohere.Client(COHERE_API_KEY)

def describe_clusters(input_csv: str, output_json: str = "cluster_descriptions.json", top_n: int = 5, model: str = "command-xlarge"):
    try:
        df = pd.read_csv(input_csv)
        if "ai_cluster" not in df.columns:
            raise ValueError("Missing 'ai_cluster' column in input CSV.")

        # Determine the content column
        text_col = None
        for c in ["quote", "data", "title"]:
            if c in df.columns:
                text_col = c
                break
        if text_col is None:
            raise ValueError("No suitable text column found in CSV. Expected one of: quote, data, title.")

        # Group by cluster
        grouped = defaultdict(list)
        for _, row in df.iterrows():
            grouped[int(row["ai_cluster"])].append(row[text_col])

        summaries = {}

        for cluster_id, texts in grouped.items():
            examples = texts[:top_n]
            # Build prompt
            prompt = (
                "You are an expert data analyst. Summarize the following items and describe the common theme or pattern they represent in 3-4 sentences:\n\n"
            )
            prompt += "\n".join([f"{i+1}. {t}" for i, t in enumerate(examples)])

            info(f"Sending cluster {cluster_id} (top {len(examples)}) to Cohere...")

            # Call Cohere
            response = co.generate(
                model=model,
                prompt=prompt,
                temperature=0.5
            )
            summary = response.generations[0].text.strip()
            summaries[str(cluster_id)] = summary
            info(f"Cluster {cluster_id} summary: {summary}")

        # Save JSON
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)
        success(f"Cluster descriptions saved to {output_json}")

    except Exception as e:
        error(f"Cohere cluster description failed: {e}")
        raise
