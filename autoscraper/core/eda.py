import pandas as pd
import json
from autoscraper.utils.logger import info, success, error

def run_eda(input_csv: str, cleaned_csv: str = "output_cleaned.csv", summary_json: str = "insights.json"):
    """
    Run basic EDA & cleaning on the scraped CSV:
    - Remove duplicates
    - Handle missing values
    - Group by predicted_categories
    - Save cleaned CSV & JSON summary
    """
    try:
        info(f"Loading scraped data from {input_csv}")
        df = pd.read_csv(input_csv)

        info("Initial rows: " + str(len(df)))
        # Drop duplicates
        df.drop_duplicates(inplace=True)

        # Fill missing values (simple approach)
        df.fillna("", inplace=True)

        # Normalize predicted_categories column
        if "predicted_categories" in df.columns:
            # Convert stringified lists to Python lists (if needed)
            def parse_tags(val):
                if pd.isna(val): return []
                if isinstance(val, list): return val
                try:
                    # Try to evaluate as JSON or Python list-like
                    parsed = json.loads(val)
                    if isinstance(parsed, list): return parsed
                    return [str(parsed)]
                except Exception:
                    # Fallback: split by commas
                    return [tag.strip() for tag in str(val).split(",") if tag.strip()]

            df["predicted_categories"] = df["predicted_categories"].apply(parse_tags)
        else:
            df["predicted_categories"] = [[] for _ in range(len(df))]

        # Save cleaned CSV
        df.to_csv(cleaned_csv, index=False)
        success(f"Cleaned data saved to {cleaned_csv}")

        # Generate summary: count of each category
        category_counts = {}
        for tags in df["predicted_categories"]:
            for tag in tags:
                category_counts[tag] = category_counts.get(tag, 0) + 1

        # Save JSON summary
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(category_counts, f, indent=2, ensure_ascii=False)
        success(f"Insights saved to {summary_json}")

        info("Top categories:")
        for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            info(f"{cat}: {count}")

    except Exception as e:
        error(f"EDA failed: {e}")
        raise
