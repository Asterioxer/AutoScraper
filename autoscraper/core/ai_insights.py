import pandas as pd
import json
from autoscraper.utils.logger import info, success, error
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# Load a lightweight embedding model once
_model = SentenceTransformer('all-MiniLM-L6-v2')

def run_ai_insights(input_csv: str, output_json: str = "ai_insights.json", clusters: int = 5):
    """
    Generate AI-driven clustering insights from scraped data.
    - Uses sentence-transformers to embed text.
    - KMeans clustering to group similar items.
    - Outputs updated CSV with cluster labels and a JSON summary.
    """
    try:
        info(f"Loading cleaned data from {input_csv}")
        df = pd.read_csv(input_csv)

        # pick the column to analyze
        text_col = None
        for c in ["quote", "data", "title"]:
            if c in df.columns:
                text_col = c
                break
        if text_col is None:
            raise ValueError("No suitable text column found in CSV.")

        texts = df[text_col].astype(str).tolist()
        info(f"Encoding {len(texts)} items with sentence-transformers...")
        embeddings = _model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

        info(f"Clustering into {clusters} groups...")
        km = KMeans(n_clusters=clusters, random_state=42, n_init='auto')
        labels = km.fit_predict(embeddings)

        df["ai_cluster"] = labels
        output_csv = "output_ai_tagged.csv"
        df.to_csv(output_csv, index=False)
        success(f"AI-tagged CSV saved to {output_csv}")

        # cluster distribution
        cluster_counts = {}
        for lbl in labels:
            cluster_counts[int(lbl)] = cluster_counts.get(int(lbl), 0) + 1

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(cluster_counts, f, indent=2)
        success(f"AI insights JSON saved to {output_json}")

        info("Cluster distribution:")
        for k, v in sorted(cluster_counts.items()):
            info(f"Cluster {k}: {v} items")

    except Exception as e:
        error(f"AI Insights failed: {e}")
        raise
