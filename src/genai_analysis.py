"""
genai_analysis.py
Handles sentiment classification, keyword extraction, and Q&A about the
review dataset using the Hugging Face Inference API (free tier, token-based).

Hugging Face setup:
  1. Create a token at https://huggingface.co/settings/tokens (a "Read" token
     with "Inference" permission is enough for the free Inference Providers API).
  2. Add it to `.streamlit/secrets.toml` as HF_TOKEN = "hf_..." (locally) or
     in Streamlit Community Cloud's Secrets settings (when deployed).

Models used (swap these constants if a model becomes unavailable on your
Hugging Face plan/provider):
  - SENTIMENT_MODEL: a dedicated 3-class sentiment classifier
  - CHAT_MODEL: an instruction-tuned chat model for the "Ask the Data" chat bot
"""

import re
import time
from collections import Counter
from typing import List, Dict

import pandas as pd
import streamlit as st
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
CHAT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SENTIMENT_PROVIDER = "hf-inference"  # this classifier is hosted on HF's own infra
CHAT_PROVIDER = "auto"  # let HF route to whichever partner (Cerebras, Together, etc.) hosts this model

_LABEL_MAP = {
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
    "label_0": "negative",
    "label_1": "neutral",
    "label_2": "positive",
}

# ---------------------------------------------------------------------------
# Hugging Face client
# ---------------------------------------------------------------------------

def get_client(provider: str = SENTIMENT_PROVIDER) -> InferenceClient:
    hf_token = st.secrets.get("HF_TOKEN", None)
    if not hf_token:
        st.error(
            "No HF_TOKEN found in Streamlit secrets. Create a token at "
            "https://huggingface.co/settings/tokens and add it to "
            "`.streamlit/secrets.toml` (locally) or Streamlit Community "
            "Cloud's Secrets settings."
        )
        st.stop()
    return InferenceClient(provider=provider, api_key=hf_token)


class GenAIUnavailableError(Exception):
    """Raised when the Hugging Face API call fails (auth, quota, model down, etc)."""


# ---------------------------------------------------------------------------
# GenAI (Hugging Face)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def classify_sentiment_batch_genai(reviews: List[str]) -> List[Dict]:
    """
    Calls the Hugging Face Inference API's text-classification task once per
    review using a dedicated sentiment model, and extracts simple keywords
    locally (Hugging Face's free-tier classifiers return only a label +
    score, not keywords, so keyword extraction is done with a lightweight
    local frequency method regardless).
    """
    client = get_client(provider=SENTIMENT_PROVIDER)
    results = []

    for text in reviews:
        try:
            raw = client.text_classification(text, model=SENTIMENT_MODEL)
        except HfHubHTTPError as e:
            raise GenAIUnavailableError(
                "Hugging Face rejected the request. This usually means: the "
                "token is invalid/missing the 'Inference' permission, you've "
                "hit the free-tier rate limit, or the model is temporarily "
                "unavailable. Check your token at "
                "https://huggingface.co/settings/tokens. "
                f"Details: {e}"
            ) from e
        except Exception as e:
            raise GenAIUnavailableError(f"Hugging Face API error: {e}") from e

        top_label = max(raw, key=lambda r: r["score"])["label"].lower()
        sentiment = _LABEL_MAP.get(top_label, "neutral")

        keywords = _extract_keywords_local(text)
        results.append({"sentiment": sentiment, "keywords": keywords})

        time.sleep(0.05)  # light courtesy delay for the free tier

    return results


def _build_game_breakdown(df: pd.DataFrame) -> str:
    """
    Pre-compute per-game stats in pandas so the chat model gets exact numbers
    (which game has the most positive reviews, etc.) instead of guessing from
    a handful of sample sentences.
    """
    if df.empty or "app_name" not in df.columns:
        return "Per-game breakdown: not available."

    lines = []
    if "sentiment" in df.columns:
        g = df.groupby("app_name")["sentiment"].value_counts().unstack(fill_value=0)
        for c in ("positive", "negative", "neutral"):
            if c not in g.columns:
                g[c] = 0
        g["total"] = g[["positive", "negative", "neutral"]].sum(axis=1)
        g["pos_rate"] = g["positive"] / g["total"]
        g = g.sort_values(["positive", "pos_rate"], ascending=False)
        lines.append("Per-game sentiment (from AI sentiment analysis), sorted by number of positive reviews:")
        for name, r in g.iterrows():
            lines.append(
                f"- {name}: {int(r['positive'])} positive, {int(r['negative'])} negative, "
                f"{int(r['neutral'])} neutral ({r['pos_rate']:.0%} positive, {int(r['total'])} reviews)"
            )
    elif "voted_up" in df.columns and df["voted_up"].notna().any():
        tmp = df.dropna(subset=["voted_up"]).copy()
        tmp["up"] = tmp["voted_up"].astype(str).str.lower().isin(["true", "1"])
        g = tmp.groupby("app_name")["up"].agg(up="sum", total="count")
        g["up_rate"] = g["up"] / g["total"]
        g = g.sort_values(["up", "up_rate"], ascending=False)
        lines.append(
            "Per-game thumbs-up votes (sentiment analysis has NOT been run yet), "
            "sorted by number of thumbs-up reviews:"
        )
        for name, r in g.iterrows():
            lines.append(f"- {name}: {int(r['up'])} thumbs-up out of {int(r['total'])} reviews ({r['up_rate']:.0%})")
    else:
        counts = df["app_name"].value_counts()
        lines.append("Reviews per game (no sentiment data available yet):")
        for name, c in counts.items():
            lines.append(f"- {name}: {int(c)} reviews")
    return "\n".join(lines)


def answer_question_about_data_genai(question: str, df: pd.DataFrame) -> str:
    """RAG-lite chatbot using a Hugging Face chat/instruct model."""
    client = get_client(provider=CHAT_PROVIDER)

    total = len(df)
    pos = int((df.get("sentiment") == "positive").sum()) if "sentiment" in df else None
    neg = int((df.get("sentiment") == "negative").sum()) if "sentiment" in df else None
    sample_df = df.sample(min(25, total), random_state=1)
    if "app_name" in sample_df.columns:
        sample_reviews = [f"[{a}] {t}" for a, t in zip(sample_df["app_name"], sample_df["review_text"])]
    else:
        sample_reviews = sample_df["review_text"].tolist()

    context = f"""
Dataset summary:
- Total reviews in current view: {total}
- Positive: {pos if pos is not None else 'N/A'}
- Negative: {neg if neg is not None else 'N/A'}

{_build_game_breakdown(df)}

Sample of up to 25 reviews from the current filtered view (game name in brackets):
{chr(10).join(f"- {r}" for r in sample_reviews)}
""".strip()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful analyst answering questions about a game "
                "review dataset. Base your answer only on the summary, the "
                "per-game statistics, and the sample reviews provided. Always "
                "give a direct answer first (for example, name the game), "
                "using the per-game statistics for any question about which "
                "game is best, worst, or most reviewed. If the data is "
                "limited or sentiment analysis has not been run, still give "
                "your best answer from what is available and add one short "
                "sentence noting the limitation. Be concise and specific."
            ),
        },
        {"role": "user", "content": f"{context}\n\nQuestion: {question}"},
    ]

    try:
        response = client.chat_completion(messages, model=CHAT_MODEL, max_tokens=400)
    except HfHubHTTPError as e:
        raise GenAIUnavailableError(
            "Hugging Face rejected the chat request. Check your token's "
            "'Inference' permission, or try again in a moment (free-tier rate "
            f"limit). Details: {e}"
        ) from e
    except Exception as e:
        raise GenAIUnavailableError(f"Hugging Face API error: {e}") from e

    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Local keyword extraction
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "the", "a", "an", "is", "it", "this", "that", "and", "or", "but", "of",
    "to", "in", "on", "for", "with", "was", "were", "are", "i", "you", "my",
    "at", "as", "be", "if", "so", "not", "very", "really", "just", "its",
    "have", "has", "had", "can", "will", "would", "than", "then", "there",
    "their", "they", "them", "game", "games", "review", "played", "playing",
}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def _extract_keywords_local(text: str, top_n: int = 3) -> List[str]:
    tokens = [t for t in _tokenize(text) if t not in _STOPWORDS and len(t) > 3]
    return [w for w, _ in Counter(tokens).most_common(top_n)]


# ---------------------------------------------------------------------------
# Entry points used by app.py
# ---------------------------------------------------------------------------

def classify_sentiment_batch(reviews: List[str]) -> List[Dict]:
    return classify_sentiment_batch_genai(reviews)


def answer_question_about_data(question: str, df: pd.DataFrame) -> str:
    return answer_question_about_data_genai(question, df)
