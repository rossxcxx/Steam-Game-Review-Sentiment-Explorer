# 🎮 GlitchGauge

## System Description

GlitchGauge is a GenAI-powered web app that analyzes
Steam game reviews. It classifies each review's sentiment
(positive, negative, or neutral), pulls out the keywords that come up most,
and shows the results as interactive charts. A built-in AI chat bot in the
side panel lets you ask plain-English questions about the reviews you are
looking at.

## The Problem

Popular games collect thousands of player reviews, and reading them one by one
is slow and impractical. Developers, publishers, and researchers who want to
know what players like, what they complain about, and whether opinion is
improving or declining have no quick way to find out. Counting "thumbs up" and
"thumbs down" votes alone hides the reasons behind them, such as bugs,
performance, pricing, or gameplay.

## The Solution

GlitchGauge turns a large pile of raw reviews into something you can read at a
glance. You narrow the reviews down with filters, run a GenAI sentiment model
on a sample of them, and see the overall mood, how it differs by game, how it
changes over time, and which topics come up most. When the charts don't answer
your question, you can ask the AI chat bot directly (for example, "What do
players complain about most?").

## Features

- **Filters** by game, genre, review date range, and minimum playtime, placed
  right below the header so you can adjust them before each analysis.
- **GenAI sentiment analysis** using a Hugging Face model
  (`cardiffnlp/twitter-roberta-base-sentiment-latest`) that labels each review
  as positive, negative, or neutral.
- **Adjustable sample size**, so you control how many reviews are analyzed and
  how much of the free API tier you use.
- **Keyword extraction** that finds the most common terms in each review.
- **Interactive visualizations** (Plotly): sentiment distribution, sentiment
  by game, sentiment trend over time, and top extracted keywords.
- **Data preview and results table**, to inspect the filtered reviews and the
  analyzed reviews with their sentiment and keywords.
- **AI chat bot in the side panel** ("Ask the Data"), powered by
  `meta-llama/Llama-3.1-8B-Instruct`. It answers free-form questions using a
  summary and a sample of the current reviews.
- **Result caching**, so re-running the same analysis doesn't call the API again.

## Dataset Source

- Kaggle "Steam Reviews Dataset":
  https://www.kaggle.com/datasets/andrewmvd/steam-reviews

## How It Works

1. **Filter:** the controls below the header narrow down the reviews, and the
   **Run Sentiment Analysis** button starts the analysis.
3. **GenAI analysis:** `src/genai_analysis.py` sends each review to the
   Hugging Face Inference API for sentiment and extracts keywords locally with
   simple word-frequency analysis.
4. **Visualize:** Plotly charts display the results.
5. **Ask the Data:** the side-panel chat bot answers questions about the
   current reviews.
