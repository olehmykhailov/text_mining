import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import os
from multiprocessing import Pool, cpu_count
import logging

def association_graph(X, vectorizer: CountVectorizer, output_path, ngram_range=(2, 2)):
    ngrams = vectorizer.get_feature_names_out()
    ngram_freq = X.sum(axis=0).A1
    ngram_freq_dict = dict(zip(ngrams, ngram_freq))
    sorted_ngrams = sorted(ngram_freq_dict.items(), key=lambda x: x[1], reverse=True)[:20]

    df = pd.DataFrame(sorted_ngrams, columns=["Ngram", "Frequency"])
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Frequency", y="Ngram", data=df, palette="Blues_d", hue="Ngram")
    plt.title(f"Top {ngram_range[0]}-grams")
    plt.tight_layout()
    plt.savefig(f"{output_path}/association_graph_{ngram_range[0]}gram.png")
    plt.close()

def network_graph(X, vectorizer: CountVectorizer,  output_path, ngram_range=(2, 2)):
    ngrams = vectorizer.get_feature_names_out()
    ngram_freq = X.sum(axis=0).A1
    ngram_freq_dict = dict(zip(ngrams, ngram_freq))

    G = nx.Graph()
    for ngram, freq in ngram_freq_dict.items():
        G.add_node(ngram, size=freq)

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=[G.nodes[node]['size'] * 10 for node in G])
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos)
    plt.title(f"Network Graph of {ngram_range[0]}-grams")
    plt.tight_layout()
    plt.savefig(f"{output_path}/network_graph_{ngram_range[0]}gram.png")
    plt.close()

def word_correlation(X, vectorizer: CountVectorizer, output_path, top_n=20):
    terms = vectorizer.get_feature_names_out()

    
    word_freq = X.sum(axis=0).A1
    top_indices = word_freq.argsort()[::-1][:top_n]
    top_terms = [terms[i] for i in top_indices]
    
    
    X_top = X[:, top_indices]
    corr_matrix = (X_top.T @ X_top).toarray()
    corr_df = pd.DataFrame(corr_matrix, index=top_terms, columns=top_terms)

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_df, cmap="YlGnBu", annot=True, fmt=".0f", linewidths=0.5)
    plt.title("Word Correlation Matrix (Top Words)")
    plt.tight_layout()
    plt.savefig(f"{output_path}/word_correlation.png")
    plt.close()


def save_association_visuals(texts, output_path):
    os.makedirs(output_path, exist_ok=True)

    vectorizer = CountVectorizer(ngram_range=(2, 2))
    X = vectorizer.fit_transform(texts)
    association_graph(X, vectorizer, output_path, ngram_range=(2, 2))
    network_graph(X, vectorizer, output_path, ngram_range=(2, 2))

    vectorizer_3 = CountVectorizer(ngram_range=(3, 3))
    X_3 = vectorizer_3.fit_transform(texts)
    association_graph(X_3, vectorizer_3, output_path, ngram_range=(3, 3))
    network_graph(X_3, vectorizer_3, output_path, ngram_range=(3, 3))

    vectorizer_full = CountVectorizer(stop_words='english')
    X_full = vectorizer_full.fit_transform(texts)
    word_correlation(X_full, vectorizer_full, output_path)



def generate_association_per_version(df: pd.DataFrame, base_dir):
    for app, group in df.groupby('appVersion'):
        combined_texts = group['processed_review'].dropna().tolist()

        output_path = os.path.join(base_dir, app, "graphs")

        save_association_visuals(combined_texts, output_path)
        logging.info(msg=f"Generated association visuals for version: {app}")

def _process_version_association(app_group):
    app, group = app_group
    combined_texts = group['processed_review'].dropna().tolist()

    if combined_texts:
        output_path = os.path.join("results", app, "graphs")
        os.makedirs(output_path, exist_ok=True)
        save_association_visuals(combined_texts, output_path)
        logging.info(msg=f"Association visuals generated for app version: {app}")
    else:
        logging.warning(msg=f"No reviews for version: {app}")

def generate_association_per_version_parallel(df: pd.DataFrame, processes=None):
    """
    Generate association graphs for each version using multiprocessing.
    """
    if processes is None:
        processes = max(cpu_count() - 1, 1)

    app_groups = list(df.groupby('appVersion'))
    logging.info(msg=f"Generating association visuals in parallel with {processes} processes...")

    with Pool(processes=processes) as pool:
        pool.map(_process_version_association, app_groups)

    logging.info(msg="All association visuals generated.")