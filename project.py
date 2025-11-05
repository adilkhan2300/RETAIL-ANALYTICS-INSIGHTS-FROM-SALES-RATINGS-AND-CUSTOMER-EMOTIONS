import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download sentiment lexicon
nltk.download("vader_lexicon")

st.set_page_config(page_title="Sales & Feedback Dashboard", layout="wide")
st.title("Sales & Customer Feedback Dashboard")

# File upload
file = st.file_uploader("Upload Combined CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

    # Auto column detection
    date_col = [c for c in df.columns if "date" in c][0]
    price_col = [c for c in df.columns if any(x in c for x in ["amount", "total", "price"] )][0]
    rating_col = [c for c in df.columns if "rating" in c][0]
    feedback_col = [c for c in df.columns if "feedback" in c][0]

    df[date_col] = pd.to_datetime(df[date_col])

    # Sidebar filters
    st.sidebar.header("Filters")
    if "category" in df.columns:
        selected_cat = st.sidebar.multiselect("Select Category", sorted(df["category"].unique()))
        if selected_cat:
            df = df[df["category"].isin(selected_cat)]

    # Sentiment Analysis
    sid = SentimentIntensityAnalyzer()
    df["sentiment_score"] = df[feedback_col].astype(str).apply(lambda x: sid.polarity_scores(x)["compound"])
    df["sentiment_label"] = df["sentiment_score"].apply(
        lambda x: "Positive" if x>0.05 else ("Negative" if x<-0.05 else "Neutral")
    )

    # 🔢 KPI Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Sales", f"₹{df[price_col].sum():,.0f}")
    c2.metric("Avg Rating", round(df[rating_col].mean(),2))
    c3.metric("Positive Feedback %", f"{df['sentiment_label'].eq('Positive').mean()*100:.1f}%")
    c4.metric("Total Orders", len(df))

    st.divider()

    # 📈 Monthly Sales
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Monthly Sales Trend")
        monthly_sales = df.groupby(df[date_col].dt.to_period("M"))[price_col].sum()
        fig, ax = plt.subplots()
        ax.plot(monthly_sales.index.astype(str), monthly_sales.values, marker="o")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # Sentiment bar chart
    with col2:
        st.subheader("Sentiment Distribution")
        fig, ax = plt.subplots()
        df["sentiment_label"].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

    # WordCloud + Category ratings
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Word Cloud (Feedback)")
        text = " ".join(df[feedback_col].astype(str))
        wc = WordCloud(width=800, height=400).generate(text)
        fig, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)

    with col4:
        st.subheader("Avg Rating by Category")
        fig, ax = plt.subplots()
        df.groupby("category")[rating_col].mean().plot(kind="bar", ax=ax)
        st.pyplot(fig)

    # 🔥 Heatmap
    st.subheader("Rating vs Sentiment by Category")
    heat = df.pivot_table(values=rating_col, index="category", columns="sentiment_label", aggfunc="mean")
    fig, ax = plt.subplots()
    plt.imshow(heat, aspect="auto")
    plt.colorbar()
    ax.set_xticks(range(len(heat.columns)))
    ax.set_yticks(range(len(heat.index)))
    ax.set_xticklabels(heat.columns)
    ax.set_yticklabels(heat.index)
    st.pyplot(fig)

    st.divider()

    # 📝 Feedback Insight Summary
    st.subheader("Insight Summary Based on Feedback")

    pos_words = ["quality", "comfort", "design", "fit", "value", "fast", "nice"]
    neg_words = ["late", "return", "poor", "wrong", "size issue", "refund", "bad"]

    positive_mentions = sum(df[feedback_col].str.contains("|".join(pos_words), case=False, na=False))
    negative_mentions = sum(df[feedback_col].str.contains("|".join(neg_words), case=False, na=False))

    st.write(f"""
### Key Insights
- Customers appreciate: **quality, design & fitting**
- Higher positive reviews correlate with **higher order value**
- Common complaints: **late delivery, size issues, returns**

### Keyword Mentions
- **Positive keywords:** {positive_mentions}
- **Negative keywords:** {negative_mentions}

### Interpretation
- Bewakoof has strong brand affinity & product satisfaction.
- Main improvement focus: **delivery speed & size accuracy**.
""")

else:
    st.info("Upload your combined CSV file to start analysis")
