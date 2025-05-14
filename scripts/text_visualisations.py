import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import seaborn as sns
from multiprocessing import Pool, cpu_count
import logging

def wordcloud_visual(texts, output_path):
    text = " ".join(texts)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "wordcloud.png"))
    plt.close()

def dtm_heatmap(X, vectorizer, output_path, top_n=20, binary=False, filename="dtm_heatmap.png"):
    df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    word_freq = df.sum().sort_values(ascending=False).head(top_n)
    top_terms = word_freq.index.tolist()
    dtm_top = df[top_terms]

    plt.figure(figsize=(14, min(0.5 * len(dtm_top), 12)))
    sns.heatmap(dtm_top, cmap="YlGnBu", cbar=True, linewidths=0.5)
    plt.title("Binary DTM" if binary else "DTM Heatmap (Top Words)")
    plt.xlabel("Words")
    plt.ylabel("Documents")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, filename))
    plt.close()

def ngram_bar_chart(texts, output_path, ngram_range=(2, 2), top_n=20):
    vectorizer = CountVectorizer(ngram_range=ngram_range)
    X = vectorizer.fit_transform(texts)
    ngram_freq = X.sum(axis=0).A1
    ngram_names = vectorizer.get_feature_names_out()
    ngram_freq_dict = dict(zip(ngram_names, ngram_freq))
    sorted_ngrams = sorted(ngram_freq_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]

    df = pd.DataFrame(sorted_ngrams, columns=["Ngram", "Frequency"])
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Frequency", y="Ngram", color="lightgreen")
    plt.title(f"Top {ngram_range[0]}-grams")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"top_{ngram_range[0]}grams.png"))
    plt.close()

def save_text_visualisations(texts, output_path):

    
    wordcloud_visual(texts, output_path)

    
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    dtm_heatmap(X, vectorizer, output_path, binary=False, filename="dtm_heatmap.png")

    
    bin_vectorizer = CountVectorizer(binary=True)
    X_bin = bin_vectorizer.fit_transform(texts)
    dtm_heatmap(X_bin, bin_vectorizer, output_path, binary=True, filename="binary_matrix.png")

    
    ngram_bar_chart(texts, output_path, ngram_range=(2, 2))
    ngram_bar_chart(texts, output_path, ngram_range=(3, 3))



def _process_version_text_visuals(app_group):
    app, group = app_group
    combined_texts = group['processed_review'].dropna().tolist()

    if combined_texts:

        output_path = os.path.join("results", app, "text_visuals")
        os.makedirs(output_path, exist_ok=True)
        save_text_visualisations(combined_texts, output_path)
        logging.info(msg=f"Visuals generated for app version: {app}")
    else:
        logging.warning(msg=f"No reviews for app version: {app}")

def generate_visuals_per_version_parallel(df: pd.DataFrame, processes=None):
    """
    Generate text graphs for each version using multiprocessing.
    """
    if processes is None:
        processes = max(cpu_count() - 1, 1)

    app_groups = list(df.groupby('appVersion'))
    logging.info(msg=f"Generating text visuals in parallel with {processes} processes...")

    with Pool(processes=processes) as pool:
        pool.map(_process_version_text_visuals, app_groups)

    logging.info(msg="All text visuals generated.")