"""
Steam Game Review Sentiment Explorer
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
    page_title="Steam Review Sentiment Explorer",
    page_icon="🎮",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🎮 Steam Game Review Sentiment Explorer")
st.caption(
    "Upload a dataset of Steam / app store game reviews, run GenAI-powered "
    "sentiment analysis and keyword extraction, and explore trends interactively."
)

# ---------------------------------------------------------------------------
# Sidebar: data source
# ---------------------------------------------------------------------------
st.sidebar.header("1. Dataset")
default_path = "data/steam_reviews_sample.csv"
uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV of game reviews", type=["csv"], help="Or leave empty to use the bundled sample dataset."
)

data_source = uploaded_file if uploaded_file is not None else default_path

try:
    df = load_data(data_source)
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()

st.sidebar.success(f"Loaded {len(df):,} reviews.")

# ---------------------------------------------------------------------------
# Sidebar: filters
# ---------------------------------------------------------------------------
st.sidebar.header("2. Filters")

game_options = sorted(df["app_name"].dropna().unique().tolist())
selected_games = st.sidebar.multiselect("Game(s)", game_options, default=[])

genre_options = sorted(df["genre"].dropna().unique().tolist())
selected_genres = st.sidebar.multiselect("Genre(s)", genre_options, default=[])

date_range = None
if df["review_date"].notna().any():
    min_date = df["review_date"].min().date()
    max_date = df["review_date"].max().date()
    date_range = st.sidebar.date_input("Review date range", (min_date, max_date))
    if isinstance(date_range, tuple) and len(date_range) != 2:
        date_range = None

min_playtime = None
if df["playtime_hours"].notna().any():
    min_playtime = st.sidebar.slider(
        "Minimum playtime (hours)", 0.0, float(df["playtime_hours"].max()), 0.0
    )

filtered_df = get_filtered_data(
    df,
    games=selected_games or None,
    genres=selected_genres or None,
    date_range=date_range,
    min_playtime=min_playtime,
)

st.sidebar.markdown(f"**{len(filtered_df):,}** reviews match current filters.")

# ---------------------------------------------------------------------------
# Sidebar: run analysis
# ---------------------------------------------------------------------------
st.sidebar.header("3. GenAI Analysis")
analysis_mode_label = st.sidebar.radio(
    "Analysis mode",
    ["GenAI (Hugging Face API)", "Offline (free demo, no API)"],
    index=0,
    help=(
        "GenAI mode requires a Hugging Face access token (free to create). "
        "Offline mode uses a local keyword-based classifier so you can test "
        "the app with no token/setup at all."
    ),
)
analysis_mode = "genai" if analysis_mode_label.startswith("GenAI") else "offline"

n_available = len(filtered_df)

if n_available == 0:
    st.sidebar.warning("No reviews match the current filters — adjust filters above to enable analysis.")
    sample_cap = 0
    run_analysis = False
    st.sidebar.button("Run Sentiment Analysis", type="primary", disabled=True)
elif n_available <= 20:
    st.sidebar.caption(f"Only {n_available} review(s) match filters — analyzing all of them.")
    sample_cap = n_available
    run_analysis = st.sidebar.button("Run Sentiment Analysis", type="primary")
else:
    slider_max = min(500, n_available)
    default_val = min(150, slider_max)
    sample_cap = st.sidebar.slider(
        "Max reviews to analyze (controls API cost)", 20, slider_max, default_val
    )
    run_analysis = st.sidebar.button("Run Sentiment Analysis", type="primary")

if "analyzed_df" not in st.session_state:
    st.session_state.analyzed_df = None
if "analysis_mode" not in st.session_state:
    st.session_state.analysis_mode = analysis_mode

if run_analysis:
    if filtered_df.empty:
        st.warning("No reviews match the current filters.")
    else:
        spinner_msg = (
            "Calling Hugging Face API to analyze reviews..."
            if analysis_mode == "genai"
            else "Running free offline keyword-based analysis..."
        )
        with st.spinner(spinner_msg):
            work_df = filtered_df.sample(
                min(sample_cap, len(filtered_df)), random_state=1
            ).reset_index(drop=True)
            try:
                results = classify_sentiment_batch(
                    work_df["review_text"].tolist(), mode=analysis_mode
                )
            except GenAIUnavailableError as e:
                st.error(str(e))
                st.info(
                    "Tip: switch 'Analysis mode' to 'Offline (free demo, no API)' "
                    "in the sidebar to keep testing without spending API credits."
                )
                st.stop()
            work_df["sentiment"] = [r.get("sentiment", "neutral") for r in results]
            work_df["keywords"] = [", ".join(r.get("keywords", [])) for r in results]
            st.session_state.analyzed_df = work_df
            st.session_state.analysis_mode = analysis_mode
        mode_note = "GenAI" if analysis_mode == "genai" else "offline (free demo)"
        st.success(f"Analyzed {len(work_df):,} reviews using {mode_note} mode.")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
tab_overview, tab_viz, tab_chat = st.tabs(["📄 Data Preview", "📊 Visualizations", "💬 Ask the Data"])

with tab_overview:
    st.subheader("Filtered Review Sample")
    st.dataframe(filtered_df.head(50), use_container_width=True)

with tab_viz:
    if st.session_state.analyzed_df is None:
        st.info("Run sentiment analysis from the sidebar to see visualizations here.")
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

with tab_chat:
    st.subheader("Ask questions about the reviews")
    st.caption(
        "Ask things like 'What do players complain about most?' or "
        "'Is sentiment trending up or down?' Answers use the currently "
        "analyzed/filtered reviews as context."
    )
    context_df = st.session_state.analyzed_df if st.session_state.analyzed_df is not None else filtered_df
    chat_mode = st.session_state.get("analysis_mode", analysis_mode)
    st.caption(f"Currently answering in **{'GenAI' if chat_mode == 'genai' else 'offline (free demo)'}** mode.")
    user_question = st.text_input("Your question")
    if st.button("Ask") and user_question.strip():
        with st.spinner("Thinking..."):
            try:
                answer = answer_question_about_data(user_question, context_df, mode=chat_mode)
            except GenAIUnavailableError as e:
                st.error(str(e))
                st.info(
                    "Tip: switch 'Analysis mode' to 'Offline (free demo, no API)' "
                    "in the sidebar to keep testing without spending API credits."
                )
                st.stop()
        st.markdown(f"**Answer:** {answer}")

st.divider()
st.caption(
    "Built with Streamlit + Hugging Face for a GenAI dataset-analysis assignment. "
    "Swap in your own Steam reviews CSV via the sidebar uploader."
)
