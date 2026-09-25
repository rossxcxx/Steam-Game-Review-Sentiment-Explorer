"""
data_loader.py
Loads and cleans the Steam/App Store game reviews dataset.
"""

import pandas as pd
import streamlit as st


REQUIRED_COLUMNS = ["review_text"]


@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    """
    Load a CSV of game reviews into a cleaned DataFrame.

    Expected (flexible) columns:
        review_id, app_name, genre, review_text, voted_up,
        playtime_hours, review_date, helpful_votes

    The loader is forgiving: it will still work if some optional
    columns are missing, but 'review_text' must be present.
    """
    df = pd.read_csv(file)

    # Normalize column names (strip spaces, lowercase, underscores)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required column(s): {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # Drop rows with empty review text
    df = df.dropna(subset=["review_text"])
    df["review_text"] = df["review_text"].astype(str).str.strip()
    df = df[df["review_text"].str.len() > 0]

    # Deduplicate identical reviews
    df = df.drop_duplicates(subset=["review_text"])

    # Fill in optional columns if missing, so the rest of the app can rely on them
    if "app_name" not in df.columns:
        df["app_name"] = "Unknown Game"
    if "genre" not in df.columns:
        df["genre"] = "Unknown"
    if "voted_up" not in df.columns:
        df["voted_up"] = None
    if "playtime_hours" not in df.columns:
        df["playtime_hours"] = None
    if "helpful_votes" not in df.columns:
        df["helpful_votes"] = 0

    # Parse dates if present
    if "review_date" in df.columns:
        df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    else:
        df["review_date"] = pd.NaT

    df = df.reset_index(drop=True)
    return df


def get_filtered_data(
    df: pd.DataFrame,
    games: list | None = None,
    genres: list | None = None,
    date_range: tuple | None = None,
    min_playtime: float | None = None,
) -> pd.DataFrame:
    """Apply sidebar filters to the cleaned dataframe."""
    filtered = df.copy()

    if games:
        filtered = filtered[filtered["app_name"].isin(games)]

    if genres:
        filtered = filtered[filtered["genre"].isin(genres)]

    if date_range and filtered["review_date"].notna().any():
        start, end = date_range
        filtered = filtered[
            (filtered["review_date"] >= pd.Timestamp(start))
            & (filtered["review_date"] <= pd.Timestamp(end))
        ]

    if min_playtime is not None and filtered["playtime_hours"].notna().any():
        filtered = filtered[filtered["playtime_hours"] >= min_playtime]

    return filtered
