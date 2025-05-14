import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from scripts.dimensionality_utils import reduce_dim_pca
import logging

def get_cluster_keywords(X, vectorizer, labels, n_clusters, n_words=5):
    feature_names = np.array(vectorizer.get_feature_names_out())
    cluster_keywords = []

    for cluster_id in range(n_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_matrix = X[cluster_indices]
        sum_words = cluster_matrix.sum(axis=0).A1
        sorted_word_indices = sum_words.argsort()[::-1][:n_words]
        top_words = feature_names[sorted_word_indices]
        cluster_keywords.append(", ".join(top_words))

    return cluster_keywords


def cluster_texts(texts, output_path, n_clusters=3):
    if len(texts) < n_clusters:
        logging.warning(msg=f"Not enough documents to form {n_clusters} clusters.")
        return

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)

    
    reduced = reduce_dim_pca(X)

    cluster_keywords = get_cluster_keywords(X, vectorizer, labels, n_clusters)

    df_vis = pd.DataFrame(reduced, columns=["PC1", "PC2"])
    df_vis["Cluster"] = labels
    df_vis["Cluster_Name"] = df_vis["Cluster"].apply(lambda x: cluster_keywords[x])


    plt.figure(figsize=(8, 6))
    for cluster_id in range(n_clusters):
        cluster_data = df_vis[df_vis["Cluster"] == cluster_id]
        plt.scatter(cluster_data["PC1"], cluster_data["PC2"], label=f"{cluster_keywords[cluster_id]}")
    
    plt.title("Text Clustering (TF-IDF + KMeans)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Clusters")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "clusterisation.png"))
    plt.close()


    df_clusters = pd.DataFrame({
        "text": texts,
        "cluster": labels,
        "cluster_name": df_vis["Cluster_Name"]
    })
    df_clusters.to_csv(os.path.join(output_path, "cluster_labels.csv"), index=False)


def generate_cluster_visuals_per_version(df: pd.DataFrame, base_dir):
    for version, group in df.groupby('appVersion'):
        combined_texts = group['processed_review'].dropna().tolist()
        output_path = os.path.join(base_dir, version, "clusterisation")
        os.makedirs(output_path, exist_ok=True)
        cluster_texts(combined_texts, output_path, n_clusters=3)
