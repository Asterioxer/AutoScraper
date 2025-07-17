import pandas as pd
from autoscraper.utils.logger import info, success, error
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load model once
_model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_enrich(input_csv: str, output_csv: str = "output_enriched.csv", sim_threshold: float = 0.90):
    """
    Phase 6: Semantic-level enrichment & deduplication
    - Normalizes text
    - Merges semantically similar rows
    """
    try:
        info(f"Loading data from {input_csv}...")
        df = pd.read_csv(input_csv)

        # Pick main text column
        text_col = None
        for c in ["quote", "data", "title"]:
            if c in df.columns:
                text_col = c
                break
        if text_col is None:
            raise ValueError("No suitable text column (quote/data/title) found.")

        texts = df[text_col].astype(str).tolist()
        info(f"Encoding {len(texts)} items for semantic comparison...")
        embeddings = _model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

        # Semantic deduplication
        keep_indices = []
        seen = np.zeros(len(texts), dtype=bool)
        for i in range(len(texts)):
            if seen[i]:
                continue
            keep_indices.append(i)
            sims = cosine_similarity([embeddings[i]], embeddings)[0]
            dup_indices = np.where(sims > sim_threshold)[0]
            for j in dup_indices:
                if j != i:
                    seen[j] = True

        enriched_df = df.iloc[keep_indices].reset_index(drop=True)
        enriched_df.to_csv(output_csv, index=False)
        success(f"Enriched CSV saved to {output_csv} (from {len(df)} → {len(enriched_df)} rows)")

    except Exception as e:
        error(f"Semantic enrichment failed: {e}")
        raise
