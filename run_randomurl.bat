:: run_leetcode_scrape.bat
python -m autoscraper.randomurl_cli ^
  --url "https://leetcode.com/tag/dynamic-programming/" ^
  --selector ".h-5" ^
  --max-pages 1 ^
  --sim-threshold 0.9 ^
  --clusters 5 ^
  --top-n 5 ^
  --model "command-xlarge"
pause
