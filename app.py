"""
GlitchGauge
A GenAI-powered Streamlit app for analyzing Steam / app store game reviews.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import load_data, get_filtered_data
from src.genai_analysis import (
    classify_sentiment_batch,
    answer_question_about_data,
    GenAIUnavailableError,
)

st.set_page_config(
    page_title="GlitchGauge",
    page_icon="🎮",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🎮 GlitchGauge")
st.caption(
    "Gauge how players really feel: run GenAI-powered sentiment analysis and "
    "keyword extraction on Steam / app store game reviews, and explore trends interactively."
)

# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------
data_source = "data/steam_reviews_sample.csv"

try:
    df = load_data(data_source)
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()

st.success(f"Loaded {len(df):,} reviews.")

# ---------------------------------------------------------------------------
# Filters (below the header)
# ---------------------------------------------------------------------------
st.subheader("Filters")

col_game, col_genre, col_date, col_play = st.columns(4)

with col_game:
    game_options = sorted(df["app_name"].dropna().unique().tolist())
    selected_games = st.multiselect("Game(s)", game_options, default=[])

with col_genre:
    genre_options = sorted(df["genre"].dropna().unique().tolist())
    selected_genres = st.multiselect("Genre(s)", genre_options, default=[])

date_range = None
with col_date:
    if df["review_date"].notna().any():
        min_date = df["review_date"].min().date()
        max_date = df["review_date"].max().date()
        date_range = st.date_input("Review date range", (min_date, max_date))
        if isinstance(date_range, tuple) and len(date_range) != 2:
            date_range = None

min_playtime = None
with col_play:
    if df["playtime_hours"].notna().any():
        min_playtime = st.slider(
            "Minimum playtime (hours)", 0.0, float(df["playtime_hours"].max()), 0.0
        )

filtered_df = get_filtered_data(
    df,
    games=selected_games or None,
    genres=selected_genres or None,
    date_range=date_range,
    min_playtime=min_playtime,
)

st.markdown(f"**{len(filtered_df):,}** reviews match current filters.")

# ---------------------------------------------------------------------------
# Run analysis (next to the filters)
# ---------------------------------------------------------------------------
n_available = len(filtered_df)

if n_available == 0:
    st.warning("No reviews match the current filters — adjust filters above to enable analysis.")
    sample_cap = 0
    run_analysis = False
    st.button("Run Sentiment Analysis", type="primary", disabled=True)
elif n_available <= 20:
    st.caption(f"Only {n_available} review(s) match filters — analyzing all of them.")
    sample_cap = n_available
    run_analysis = st.button("Run Sentiment Analysis", type="primary")
else:
    slider_max = min(500, n_available)
    default_val = min(150, slider_max)
    sample_cap = st.slider(
        "Max reviews to analyze (controls API cost)", 20, slider_max, default_val
    )
    run_analysis = st.button("Run Sentiment Analysis", type="primary")

if "analyzed_df" not in st.session_state:
    st.session_state.analyzed_df = None

if run_analysis:
    if filtered_df.empty:
        st.warning("No reviews match the current filters.")
    else:
        with st.spinner("Calling Hugging Face API to analyze reviews..."):
            work_df = filtered_df.sample(
                min(sample_cap, len(filtered_df)), random_state=1
            ).reset_index(drop=True)
            try:
                results = classify_sentiment_batch(work_df["review_text"].tolist())
            except GenAIUnavailableError as e:
                st.error(str(e))
                st.stop()
            work_df["sentiment"] = [r.get("sentiment", "neutral") for r in results]
            work_df["keywords"] = [", ".join(r.get("keywords", [])) for r in results]
            st.session_state.analyzed_df = work_df
        st.success(f"Analyzed {len(work_df):,} reviews using GenAI.")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
tab_overview, tab_viz = st.tabs(["📄 Data Preview", "📊 Visualizations"])

with tab_overview:
    st.subheader("Filtered Review Sample")
    st.dataframe(filtered_df.head(50), use_container_width=True)

with tab_viz:
    if st.session_state.analyzed_df is None:
        st.info("Run sentiment analysis above to see visualizations here.")
    else:
        adf = st.session_state.analyzed_df
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Sentiment Distribution")
            counts = adf["sentiment"].value_counts().reset_index()
            counts.columns = ["sentiment", "count"]
            fig = px.bar(
                counts, x="sentiment", y="count", color="sentiment",
                color_discrete_map={"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Sentiment by Game")
            by_game = adf.groupby(["app_name", "sentiment"]).size().reset_index(name="count")
            fig2 = px.bar(
                by_game, x="app_name", y="count", color="sentiment", barmode="stack",
                color_discrete_map={"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"},
            )
            fig2.update_layout(xaxis_title="", xaxis_tickangle=-30)
            st.plotly_chart(fig2, use_container_width=True)

        if adf["review_date"].notna().any():
            st.subheader("Sentiment Trend Over Time")
            trend = adf.dropna(subset=["review_date"]).copy()
            trend["month"] = trend["review_date"].dt.to_period("M").astype(str)
            trend_counts = trend.groupby(["month", "sentiment"]).size().reset_index(name="count")
            fig3 = px.line(
                trend_counts, x="month", y="count", color="sentiment", markers=True,
                color_discrete_map={"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"},
            )
            st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Top Extracted Keywords")
        kw_series = adf["keywords"].str.split(", ").explode().dropna()
        kw_series = kw_series[kw_series != ""]
        top_kw = kw_series.value_counts().head(15).reset_index()
        top_kw.columns = ["keyword", "count"]
        fig4 = px.bar(top_kw, x="count", y="keyword", orientation="h")
        fig4.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig4, use_container_width=True)

        with st.expander("See analyzed reviews with sentiment + keywords"):
            st.dataframe(
                adf[["app_name", "review_text", "sentiment", "keywords"]],
                use_container_width=True,
            )

# ---------------------------------------------------------------------------
# Sidebar: AI chat bot
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("💬 Ask the Data")
    st.caption(
        "Ask things like 'What do players complain about most?' or "
        "'Is sentiment trending up or down?' Answers use the currently "
        "analyzed/filtered reviews as context."
    )
    context_df = st.session_state.analyzed_df if st.session_state.analyzed_df is not None else filtered_df
    user_question = st.text_input("Your question")
    if st.button("Ask") and user_question.strip():
        with st.spinner("Thinking..."):
            try:
                answer = answer_question_about_data(user_question, context_df)
            except GenAIUnavailableError as e:
                st.error(str(e))
                st.stop()
        st.markdown(f"**Answer:** {answer}")

st.divider()
st.caption("Built with Streamlit + Hugging Face for a GenAI dataset-analysis assignment.")
