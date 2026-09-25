# 🎮 Steam Game Review Sentiment Explorer

A GenAI-powered Streamlit app that analyzes Steam / app store game reviews for
sentiment, keywords/themes, and trends — with a built-in chatbot for asking
questions about the dataset.

## Project Structure

```
steam_review_sentiment_app/
├── app.py                          # Main Streamlit app
├── requirements.txt                # Python dependencies
├── README.md
├── data/
│   └── steam_reviews_sample.csv    # Bundled SAMPLE dataset (synthetic, for demo/testing)
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Loading + cleaning logic
│   └── genai_analysis.py           # Hugging Face API calls: sentiment, keywords, Q&A
└── .streamlit/
    └── secrets_example.toml        # Template for your API key (rename to secrets.toml)
```

## ⚠️ About the bundled dataset

`data/steam_reviews_sample.csv` is a **synthetically generated sample** (470
rows across 10 fictional games) that mirrors the structure of real Steam
review datasets, so you can run and test the app immediately without any
setup. **Before your final submission/deployment, swap in a real dataset**,
for example:

- Kaggle "Steam Reviews Dataset": https://www.kaggle.com/datasets/andrewmvd/steam-reviews
- Kaggle "Steam Reviews" (100M+ reviews, sample it down): search "steam reviews" on Kaggle

Download the CSV, rename/place it at `data/steam_reviews.csv`, and either:
- Update `default_path` in `app.py` to point to it, or
- Just use the app's sidebar **file uploader** to load it at runtime (no code change needed).

The loader in `src/data_loader.py` is flexible about column names — it only
strictly requires a `review_text` column. If your real dataset uses different
column names (e.g. `review`, `content`), rename them to match:

| Expected column   | Description                          |
|-------------------|---------------------------------------|
| `review_text`     | The review text (required)            |
| `app_name`        | Game/app name                         |
| `genre`           | Genre/category                        |
| `voted_up`        | Boolean: recommended or not           |
| `playtime_hours`  | Hours played at time of review        |
| `review_date`     | Date of the review (YYYY-MM-DD)       |
| `helpful_votes`   | Number of helpful votes               |

## Setup (local)

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add your Hugging Face access token:
   ```bash
   cp .streamlit/secrets_example.toml .streamlit/secrets.toml
   # then edit .streamlit/secrets.toml and paste your real token
   ```
   Get a free token at https://huggingface.co/settings/tokens — click "New
   token", choose type "Read" (or "Fine-grained" with the "Make calls to
   Inference Providers" permission enabled), and copy the value starting
   with `hf_`.

4. Run the app:
   ```bash
   streamlit run app.py
   ```

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

## Deploying to Streamlit Community Cloud

1. Push this folder to a GitHub repository (make sure `.streamlit/secrets.toml`
   is in your `.gitignore` — never commit real tokens).
2. Go to https://share.streamlit.io, sign in, and click "New app".
3. Point it at your repo, branch, and `app.py` as the entry point.
4. In **App settings → Secrets**, paste:
   ```toml
   HF_TOKEN = "hf_your-real-token"
   ```
5. Deploy. Your app will be live at a `*.streamlit.app` URL.

## Troubleshooting

- **"No HF_TOKEN found"** — you haven't created `.streamlit/secrets.toml`
  locally (or added the secret in Streamlit Cloud) yet. See Setup step 3.
- **401/403 errors** — your token may lack the "Inference" permission.
  Regenerate it at https://huggingface.co/settings/tokens and make sure
  "Make calls to Inference Providers" is checked.
- **429 / rate limited** — the free Inference API tier has usage limits per
  hour. Wait a bit, lower "Max reviews to analyze" in the sidebar, or switch
  to Offline mode to keep working in the meantime.
- **Model unavailable / cold start errors** — hosted free models occasionally
  scale to zero and take a few seconds to "warm up" on first call. The chat
  model uses `provider="auto"` so Hugging Face routes to whichever partner
  (Cerebras, Together, SambaNova, etc.) currently hosts it — if `CHAT_MODEL`
  in `src/genai_analysis.py` ever stops working, swap it for another
  instruction-tuned chat model listed at
  https://huggingface.co/models?inference_provider=all&other=conversational&sort=trending.
  The sentiment model is pinned to the `hf-inference` provider specifically,
  since lightweight classifiers like it are usually only hosted on
  Hugging Face's own infrastructure rather than third-party providers.

## Next Goals (per assignment)

- ✅ Filters by genre and game are already included in the sidebar.
- ✅ A basic "Ask the Data" chatbot tab is included.
- Ideas to extend further:
  - Add a genre/game leaderboard ranking by average sentiment.
  - Cache GenAI results to disk/DB so re-deploys don't re-analyze from scratch.
  - Add support for OpenAI or Anthropic's Claude API as an alternative model provider.
  - Add review-length or helpful-votes weighting to sentiment aggregation.
