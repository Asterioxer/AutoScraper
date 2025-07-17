from autoscraper.core.scraper import scrape

def test_scraper_basic():
    data = scrape("https://quotes.toscrape.com", ".text")
    assert len(data) > 0
