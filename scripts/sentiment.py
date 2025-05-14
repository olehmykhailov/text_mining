from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):

    if not isinstance(text, str) or text.strip() == "":
        return None

    score = analyzer.polarity_scores(text)
    return score["compound"]

def label_sentiment(score):
    
    if score is None:
        return "neutral"
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

def apply_sentiment_analysis(df: pd.DataFrame, text_column="processed_review"):

    df["sentiment_score"] = df[text_column].apply(analyze_sentiment)
    df["sentiment_label"] = df["sentiment_score"].apply(label_sentiment)
    return df

def visualize_sentiment_per_version(df, output_dir):
    
    apply_sentiment_analysis(df)

    grouped = df.groupby("appVersion")

    for version, group in grouped:
        sentiment_counts = group["sentiment_label"].value_counts().reindex(["positive", "neutral", "negative"], fill_value=0)
        
        
        plt.figure(figsize=(6, 4))
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="coolwarm")
        plt.title(f"Sentiment distribution: {version}")
        plt.xlabel("Sentiment")
        plt.ylabel("Count")
        plt.tight_layout()

        

        version_path = os.path.join(output_dir, version, "sentiment", "sentiment.png")
        plt.savefig(version_path)
        plt.close()
