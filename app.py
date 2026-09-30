"""
GlitchGauge
A GenAI-powered Streamlit app for analyzing Steam / app store game reviews.
"""

import html as _html

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
# Look & feel (display only)
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&display=swap');

:root {
    --gg-ink: #15131F;
    --gg-accent: #0E9F9A;
    --gg-pos: #2ecc71;
    --gg-neg: #e74c3c;
    --gg-neu: #95a5a6;
    --gg-line: rgba(128, 128, 128, 0.28);
    --gg-soft: rgba(128, 128, 128, 0.08);
    --gg-display: 'Space Grotesk', system-ui, -apple-system, 'Segoe UI', sans-serif;
}

.block-container { padding-top: 2rem; max-width: 1400px; }

h2, h3, h4 { font-family: var(--gg-display) !important; letter-spacing: -0.01em; }

/* Hero */
.gg-hero {
    background: var(--gg-ink);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 1.25rem;
}
.gg-title {
    font-family: var(--gg-display);
    font-weight: 700;
    font-size: 2.7rem;
    line-height: 1.1;
    letter-spacing: -0.02em;
    color: #F4F2FA;
    text-shadow: -2px 0 rgba(255, 61, 129, 0.75), 2px 0 rgba(25, 195, 212, 0.75);
}
.gg-tagline {
    color: #B9B4CC;
    margin: 0.6rem 0 0;
    font-size: 1.02rem;
    max-width: 64ch;
    line-height: 1.5;
}

/* Metric tiles */
[data-testid="stMetric"] {
    background: var(--gg-soft);
    border: 1px solid var(--gg-line);
    border-radius: 12px;
    padding: 14px 18px;
}
[data-testid="stMetricValue"] { font-family: var(--gg-display); font-weight: 700; }

/* Primary button */
button[kind="primary"], [data-testid="stBaseButton-primary"] {
    background: var(--gg-accent);
    border: none;
    color: #fff;
    font-weight: 600;
    border-radius: 10px;
    padding: 0.55rem 1rem;
}
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {
    filter: brightness(1.1);
    color: #fff;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
.stTabs [data-baseweb="tab"] { font-weight: 600; padding: 0.55rem 1rem; }

/* Sidebar */
[data-testid="stSidebar"] { border-right: 1px solid var(--gg-line); }

/* Review cards */
.gg-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(290px, 1fr));
    gap: 14px;
    margin-top: 0.5rem;
}
.gg-card {
    display: flex;
    flex-direction: column;
    gap: 10px;
    min-height: 190px;
    padding: 16px 18px;
    border: 1px solid var(--gg-line);
    border-left: 6px solid var(--gg-neu);
    border-radius: 12px;
    background: var(--gg-soft);
}
.gg-card.pos { border-left-color: var(--gg-pos); }
.gg-card.neg { border-left-color: var(--gg-neg); }
.gg-head { display: flex; justify-content: space-between; align-items: baseline; gap: 10px; }
.gg-game { font-family: var(--gg-display); font-weight: 700; font-size: 1.08rem; }
.gg-verdict { font-size: 0.8rem; font-weight: 600; white-space: nowrap; }
.gg-card.pos .gg-verdict { color: var(--gg-pos); }
.gg-card.neg .gg-verdict { color: var(--gg-neg); }
.gg-genre { font-size: 0.85rem; opacity: 0.7; margin-top: -6px; }
.gg-text { font-size: 0.98rem; line-height: 1.55; flex: 1; }
.gg-foot { display: flex; flex-wrap: wrap; gap: 6px; }
.gg-chip {
    font-size: 0.78rem;
    padding: 2px 10px;
    border-radius: 999px;
    border: 1px solid var(--gg-line);
    opacity: 0.85;
}

/* Footer */
.gg-footer {
    text-align: center;
    opacity: 0.6;
    font-size: 0.85rem;
    margin-top: 2.5rem;
    padding-top: 1rem;
    border-top: 1px solid var(--gg-line);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def _esc(value) -> str:
    """Escape text for safe use inside the custom HTML cards."""
    return _html.escape(str(value)).replace("$", "&#36;")


def build_review_cards_html(records) -> str:
    """Build the review-card grid as a single HTML string (display only)."""
    cards = []
    for r in records:
        voted = str(r.get("voted_up")).strip().lower()
        if voted in ("true", "1"):
            tone, verdict = "pos", "👍 Recommended"
        elif voted in ("false", "0"):
            tone, verdict = "neg", "👎 Not recommended"
        else:
            tone, verdict = "", ""

        name = _esc(r.get("app_name", "Unknown Game"))
        genre = r.get("genre")
        text = _esc(r["review_text"])

        chips = []
        if pd.notna(r.get("review_date")):
            chips.append(f"📅 {r['review_date'].strftime('%b %d, %Y')}")
        if pd.notna(r.get("playtime_hours")):
            chips.append(f"⏱️ {float(r['playtime_hours']):,.1f} hrs")
        if pd.notna(r.get("helpful_votes")):
            chips.append(f"🙌 {int(r['helpful_votes']):,} helpful")

        parts = [f'<div class="gg-card {tone}">']
        parts.append(
            f'<div class="gg-head"><div class="gg-game">{name}</div>'
            + (f'<div class="gg-verdict">{verdict}</div>' if verdict else "")
            + "</div>"
        )
        if pd.notna(genre) and str(genre).strip():
            parts.append(f'<div class="gg-genre">{_esc(genre)}</div>')
        parts.append(f'<div class="gg-text">{text}</div>')
        if chips:
            parts.append(
                '<div class="gg-foot">'
                + "".join(f'<span class="gg-chip">{_esc(c)}</span>' for c in chips)
                + "</div>"
            )
        parts.append("</div>")
        cards.append("".join(parts))
    return '<div class="gg-grid">' + "".join(cards) + "</div>"


def style_fig(fig, height=340):
    """Consistent, theme-friendly chart styling (display only)."""
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        bargap=0.3,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)")
    return fig


SENTIMENT_ORDER = {"sentiment": ["positive", "neutral", "negative"]}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="gg-hero">'
    '<div class="gg-title">🎮 GlitchGauge</div>'
    '<p class="gg-tagline">Gauge how players really feel: run GenAI-powered sentiment analysis and '
    "keyword extraction on Steam / app store game reviews, and explore trends interactively.</p>"
    "</div>",
    unsafe_allow_html=True,
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

# ---------------------------------------------------------------------------
# Filters (below the header)
# ---------------------------------------------------------------------------
with st.container(border=True):
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

m1, m2, m3 = st.columns(3)
m1.metric("Reviews loaded", f"{len(df):,}")
m2.metric("Match your filters", f"{len(filtered_df):,}")
m3.metric("Games in view", f"{filtered_df['app_name'].nunique():,}")

# ---------------------------------------------------------------------------
# Run analysis (next to the filters)
# ---------------------------------------------------------------------------
n_available = len(filtered_df)

with st.container(border=True):
    st.subheader("Run analysis")

    if n_available == 0:
        st.warning("No reviews match the current filters — adjust filters above to enable analysis.")
        sample_cap = 0
        run_analysis = False
        st.button("Run Sentiment Analysis", type="primary", disabled=True, use_container_width=True)
    elif n_available <= 20:
        st.caption(f"Only {n_available} review(s) match filters — analyzing all of them.")
        sample_cap = n_available
        run_analysis = st.button("Run Sentiment Analysis", type="primary", use_container_width=True)
    else:
        slider_max = min(500, n_available)
        default_val = min(150, slider_max)
        sample_cap = st.slider(
            "Max reviews to analyze (controls API cost)", 20, slider_max, default_val
        )
        run_analysis = st.button("Run Sentiment Analysis", type="primary", use_container_width=True)

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
    preview_df = filtered_df.head(50)

    if preview_df.empty:
        st.info("No reviews to show. Adjust the filters above.")
    else:
        st.markdown(
            build_review_cards_html(preview_df.to_dict("records")),
            unsafe_allow_html=True,
        )

with tab_viz:
    if st.session_state.analyzed_df is None:
        st.info("Run sentiment analysis above to see visualizations here.")
    else:
        adf = st.session_state.analyzed_df

        # Quick sentiment summary tiles
        sent_counts = adf["sentiment"].value_counts()
        total_analyzed = max(len(adf), 1)
        k1, k2, k3 = st.columns(3)
        for tile, label, key in (
            (k1, "😊 Positive", "positive"),
            (k2, "😐 Neutral", "neutral"),
            (k3, "😠 Negative", "negative"),
        ):
            n = int(sent_counts.get(key, 0))
            tile.metric(label, f"{n:,} ({n / total_analyzed:.0%})")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Sentiment Distribution")
            counts = adf["sentiment"].value_counts().reset_index()
            counts.columns = ["sentiment", "count"]
            fig = px.bar(
                counts, x="sentiment", y="count", color="sentiment",
                category_orders=SENTIMENT_ORDER,
                color_discrete_map={"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"},
            )
            fig.update_layout(showlegend=False, xaxis_title="")
            st.plotly_chart(style_fig(fig), use_container_width=True)

        with col2:
            st.subheader("Sentiment by Game")
            by_game = adf.groupby(["app_name", "sentiment"]).size().reset_index(name="count")
            fig2 = px.bar(
                by_game, x="app_name", y="count", color="sentiment", barmode="stack",
                category_orders=SENTIMENT_ORDER,
                color_discrete_map={"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"},
            )
            fig2.update_layout(xaxis_title="", xaxis_tickangle=-30)
            st.plotly_chart(style_fig(fig2), use_container_width=True)

        if adf["review_date"].notna().any():
            st.subheader("Sentiment Trend Over Time")
            trend = adf.dropna(subset=["review_date"]).copy()
            trend["month"] = trend["review_date"].dt.to_period("M").astype(str)
            trend_counts = trend.groupby(["month", "sentiment"]).size().reset_index(name="count")
            fig3 = px.line(
                trend_counts, x="month", y="count", color="sentiment", markers=True,
                category_orders=SENTIMENT_ORDER,
                color_discrete_map={"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"},
            )
            st.plotly_chart(style_fig(fig3, height=360), use_container_width=True)

        st.subheader("Top Extracted Keywords")
        kw_series = adf["keywords"].str.split(", ").explode().dropna()
        kw_series = kw_series[kw_series != ""]
        top_kw = kw_series.value_counts().head(15).reset_index()
        top_kw.columns = ["keyword", "count"]
        fig4 = px.bar(
            top_kw, x="count", y="keyword", orientation="h",
            color_discrete_sequence=["#0E9F9A"],
        )
        fig4.update_layout(yaxis={"categoryorder": "total ascending"}, yaxis_title="")
        st.plotly_chart(style_fig(fig4, height=440), use_container_width=True)

        with st.expander("See analyzed reviews with sentiment + keywords"):
            st.dataframe(
                adf[["app_name", "review_text", "sentiment", "keywords"]],
                use_container_width=True,
            )

# ---------------------------------------------------------------------------
# Sidebar: AI chat bot
# ---------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("💬 Ask the Data")
    st.caption(
        "Ask about the reviews, e.g. 'What do players complain about most?' "
        "Answers use the analyzed (or filtered) reviews as context."
    )

    if st.button(
        "🗑️ Clear chat",
        use_container_width=True,
        disabled=not st.session_state.chat_history,
    ):
        st.session_state.chat_history = []
        st.rerun()

    context_df = st.session_state.analyzed_df if st.session_state.analyzed_df is not None else filtered_df

    # Scrollable message area
    chat_box = st.container(height=450, border=True)
    with chat_box:
        if not st.session_state.chat_history:
            st.caption("No messages yet. Type a question below to get started.")
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    user_question = st.chat_input("Ask about the reviews...")
    if user_question and user_question.strip():
        question = user_question.strip()
        with chat_box:
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        answer = answer_question_about_data(question, context_df)
                    except GenAIUnavailableError as e:
                        st.error(str(e))
                        st.stop()
                st.markdown(answer)
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

st.markdown(
    '<div class="gg-footer">Built with Streamlit + Hugging Face for a GenAI dataset-analysis assignment.</div>',
    unsafe_allow_html=True,
)
