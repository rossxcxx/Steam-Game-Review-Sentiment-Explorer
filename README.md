# 🎮 Steam Game Review Sentiment Explorer

A GenAI-powered Streamlit app that analyzes Steam / app store game reviews for
sentiment, keywords/themes, and trends — with a built-in chatbot for asking
questions about the dataset.


## 📊 Dataset source

- Kaggle "Steam Reviews Dataset": https://www.kaggle.com/datasets/andrewmvd/steam-reviews


## How it works

1. **Load & clean** — `src/data_loader.py` loads the CSV with Pandas, strips
   empty/duplicate reviews, and normalizes column names.
2. **Filter** — sidebar widgets let you filter by game, genre, date range, and
   minimum playtime before running analysis (keeps things fast and light on
   the free tier).
3. **GenAI analysis** — `src/genai_analysis.py` sends each review to the
   Hugging Face Inference API using the `cardiffnlp/twitter-roberta-base-sentiment-latest`
   model for sentiment (positive/negative/neutral), and extracts keywords
   locally via simple word-frequency analysis. Results are cached with
   `st.cache_data` so re-running the app doesn't re-call the API on the
   same input. A **chat model** (`HuggingFaceH4/zephyr-7b-beta` by default)
   powers the "Ask the Data" tab.
4. **Offline fallback** — if you don't have a token yet, or hit a free-tier
   rate limit, switch to "Offline (free demo, no API)" mode in the sidebar.
   It uses a local lexicon-based classifier with zero API calls, so you can
   still test the whole app end to end.
5. **Visualize** — Plotly charts show sentiment distribution, sentiment by
   game, sentiment trend over time, and top extracted keywords.
6. **Ask the Data** — a chatbot tab feeds a summary + sample of the currently
   filtered reviews to the model so you can ask free-form questions
   ("What do players complain about most?").

## Next Goals (per assignment)

- ✅ Filters by genre and game are already included in the sidebar.
- ✅ A basic "Ask the Data" chatbot tab is included.
- Ideas to extend further:
  - Add a genre/game leaderboard ranking by average sentiment.
  - Cache GenAI results to disk/DB so re-deploys don't re-analyze from scratch.
  - Add support for OpenAI or Anthropic's Claude API as an alternative model provider.
  - Add review-length or helpful-votes weighting to sentiment aggregation.
