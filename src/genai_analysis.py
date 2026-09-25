"""
genai_analysis.py
Handles sentiment classification, keyword extraction, and Q&A about the
review dataset. Supports two modes:

  - "genai"   : calls the Hugging Face Inference API (free tier, token-based)
  - "offline" : a free, local, lexicon-based fallback with no API calls at
                all, so the app is fully testable/demoable with zero setup.

Hugging Face setup:
  1. Create a token at https://huggingface.co/settings/tokens (a "Read" token
     with "Inference" permission is enough for the free Inference Providers API).
  2. Add it to `.streamlit/secrets.toml` as HF_TOKEN = "hf_..." (locally) or
     in Streamlit Community Cloud's Secrets settings (when deployed).

Models used (swap these constants if a model becomes unavailable on your
Hugging Face plan/provider):
  - SENTIMENT_MODEL: a dedicated 3-class sentiment classifier
  - CHAT_MODEL: an instruction-tuned chat model for the "Ask the Data" tab
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
            "Cloud's Secrets settings, or switch to 'Offline (free demo)' "
            "mode in the sidebar."
        )
        st.stop()
    return InferenceClient(provider=provider, api_key=hf_token)


class GenAIUnavailableError(Exception):
    """Raised when the Hugging Face API call fails (auth, quota, model down, etc)."""


# ---------------------------------------------------------------------------
# GenAI (Hugging Face) mode
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def classify_sentiment_batch_genai(reviews: List[str]) -> List[Dict]:
    """
    Calls the Hugging Face Inference API's text-classification task once per
    review using a dedicated sentiment model, and extracts simple keywords
    locally (Hugging Face's free-tier classifiers return only a label +
    score, not keywords, so keyword extraction is done with a lightweight
    local frequency method regardless of mode).
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
                "https://huggingface.co/settings/tokens, or switch to "
                "'Offline (free demo)' mode in the sidebar. "
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


def answer_question_about_data_genai(question: str, df: pd.DataFrame) -> str:
    """RAG-lite chatbot using a Hugging Face chat/instruct model."""
    client = get_client(provider=CHAT_PROVIDER)

    total = len(df)
    pos = int((df.get("sentiment") == "positive").sum()) if "sentiment" in df else None
    neg = int((df.get("sentiment") == "negative").sum()) if "sentiment" in df else None
    sample_reviews = df["review_text"].sample(min(25, total), random_state=1).tolist()

    context = f"""
Dataset summary:
- Total reviews in current view: {total}
- Positive: {pos if pos is not None else 'N/A'}
- Negative: {neg if neg is not None else 'N/A'}

Sample of up to 25 reviews from the current filtered view:
{chr(10).join(f"- {r}" for r in sample_reviews)}
""".strip()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful analyst answering questions about a game "
                "review dataset. Base your answer only on the summary and "
                "sample reviews provided. Be concise and specific."
            ),
        },
        {"role": "user", "content": f"{context}\n\nQuestion: {question}"},
    ]

    try:
        response = client.chat_completion(messages, model=CHAT_MODEL, max_tokens=400)
    except HfHubHTTPError as e:
        raise GenAIUnavailableError(
            "Hugging Face rejected the chat request. Check your token's "
            "'Inference' permission, try again in a moment (free-tier rate "
            f"limit), or switch to 'Offline (free demo)' mode. Details: {e}"
        ) from e
    except Exception as e:
        raise GenAIUnavailableError(f"Hugging Face API error: {e}") from e

    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Shared local keyword extraction (used in both genai and offline modes)
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
# Offline (free, no API) fallback mode
# ---------------------------------------------------------------------------

_POSITIVE_WORDS = {
    "love", "loved", "great", "best", "amazing", "incredible", "fun", "addictive",
    "beautiful", "polished", "recommend", "recommended", "worth", "excellent",
    "enjoy", "enjoyed", "enjoyable", "fantastic", "awesome", "solid", "smooth",
    "stunning", "impressive", "satisfying", "hooked", "gem", "masterpiece",
}
_NEGATIVE_WORDS = {
    "bug", "bugs", "broken", "crash", "crashes", "unplayable", "disappointing",
    "disappointed", "frustrating", "repetitive", "boring", "worse", "avoid",
    "issue", "issues", "unfinished", "rushed", "bad", "worst", "terrible",
    "laggy", "glitchy", "pay-to-win", "grindy", "overpriced",
}


def classify_sentiment_batch_offline(reviews: List[str]) -> List[Dict]:
    """
    Free, local, lexicon-based sentiment classifier + naive keyword extraction.
    No API calls, no token needed. Good enough for demoing the app's UI/UX,
    not a substitute for real model-quality analysis.
    """
    results = []
    for text in reviews:
        tokens = _tokenize(text)
        pos_hits = sum(1 for t in tokens if t in _POSITIVE_WORDS)
        neg_hits = sum(1 for t in tokens if t in _NEGATIVE_WORDS)

        if pos_hits > neg_hits:
            sentiment = "positive"
        elif neg_hits > pos_hits:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        results.append({"sentiment": sentiment, "keywords": _extract_keywords_local(text)})
    return results


def answer_question_about_data_offline(question: str, df: pd.DataFrame) -> str:
    """
    Free, local, rule-based 'chatbot' fallback: answers a small set of
    common question patterns using stats computed directly from the data,
    instead of calling any API.
    """
    total = len(df)
    if total == 0:
        return "There are no reviews in the current view to analyze."

    q = question.lower()
    has_sentiment = "sentiment" in df.columns

    if has_sentiment:
        counts = df["sentiment"].value_counts()
        pos = int(counts.get("positive", 0))
        neg = int(counts.get("negative", 0))
        neu = int(counts.get("neutral", 0))
    else:
        pos = neg = neu = None

    if any(w in q for w in ["complain", "negative", "problem", "issue", "bad"]):
        if "keywords" in df.columns:
            neg_kw = (
                df.loc[df.get("sentiment") == "negative", "keywords"]
                .str.split(", ")
                .explode()
                .dropna()
            )
            neg_kw = neg_kw[neg_kw != ""]
            if not neg_kw.empty:
                top = ", ".join(neg_kw.value_counts().head(5).index.tolist())
                return f"[Offline mode] Most common negative themes: {top}."
        return "[Offline mode] Run sentiment analysis first so I can identify complaint themes."

    if any(w in q for w in ["popular", "best", "top", "recommend"]):
        if "app_name" in df.columns:
            top_games = df["app_name"].value_counts().head(3)
            listing = ", ".join(f"{g} ({c} reviews)" for g, c in top_games.items())
            return f"[Offline mode] Games with the most reviews in this view: {listing}."

    if any(w in q for w in ["trend", "over time", "improving", "declining"]):
        if "review_date" in df.columns and df["review_date"].notna().any() and has_sentiment:
            trend = df.dropna(subset=["review_date"]).copy()
            trend["month"] = trend["review_date"].dt.to_period("M")
            recent = trend[trend["month"] == trend["month"].max()]
            recent_pos_rate = (recent["sentiment"] == "positive").mean() if len(recent) else None
            if recent_pos_rate is not None:
                return (
                    f"[Offline mode] In the most recent month in this view, "
                    f"{recent_pos_rate:.0%} of reviews were positive "
                    f"(out of {pos + neg + neu} total analyzed)."
                )

    if pos is not None:
        return (
            f"[Offline mode - no API used] Of {total} reviews in the current view: "
            f"{pos} positive, {neg} negative, {neu} neutral. "
            f"For richer, natural-language answers, add an HF_TOKEN and "
            f"switch to GenAI mode in the sidebar."
        )

    return (
        "[Offline mode] Run sentiment analysis first (sidebar) so I have "
        "sentiment data to answer questions about."
    )


# ---------------------------------------------------------------------------
# Unified entry points used by app.py
# ---------------------------------------------------------------------------

def classify_sentiment_batch(reviews: List[str], mode: str = "genai") -> List[Dict]:
    if mode == "offline":
        return classify_sentiment_batch_offline(reviews)
    return classify_sentiment_batch_genai(reviews)


def answer_question_about_data(question: str, df: pd.DataFrame, mode: str = "genai") -> str:
    if mode == "offline":
        return answer_question_about_data_offline(question, df)
    return answer_question_about_data_genai(question, df)
