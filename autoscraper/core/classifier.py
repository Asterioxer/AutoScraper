# autoscraper/core/classifier.py

class SimpleClassifier:
    """
    A simple rule-based classifier for scraped items.
    Extend or replace with ML later.
    """
    def classify(self, item: dict) -> list[str]:
        tags = []
        combined = " ".join(str(v).lower() for v in item.values() if v)
        if "life" in combined:
            tags.append("life")
        if "truth" in combined:
            tags.append("truth")
        if "love" in combined:
            tags.append("love")
        return tags
