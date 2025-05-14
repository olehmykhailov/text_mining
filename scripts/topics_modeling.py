from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import os
import logging

def lda_topic_modeling(texts, n_topics=5, n_words=10):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    
    feature_names = vectorizer.get_feature_names_out()
    
    topics = []
    for idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]]
        topics.append(f"Topic {idx + 1}: {', '.join(top_words)}")
    
    return topics

def save_lda_topics(texts, output_path, n_topics=5, n_words=10):
    
    
    topics = lda_topic_modeling(texts, n_topics=n_topics, n_words=n_words)
    
    with open(os.path.join(output_path, "lda_topics.txt"), "w") as f:
        for topic in topics:
            f.write(topic + "\n")
    
    logging.info(f"LDA topics saved to {os.path.join(output_path, 'lda_topics.txt')}")

def generate_lda_topics_per_version(df, base_dir, n_topics=5, n_words=10):
    for version, group in df.groupby("appVersion"):
        texts = group["processed_review"].dropna().tolist()

        output_path = os.path.join(base_dir, version, "topics")

        os.makedirs(output_path, exist_ok=True)

        if texts:
            save_lda_topics(texts, output_path, n_topics=n_topics, n_words=n_words)
