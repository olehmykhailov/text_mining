from scripts.utilities import (
    download_nltk_resources,
    create_version_directories,
    clear_output_dir
)
from scripts.data_ops import preprocess_data, save_data
from scripts.text_visualisations import generate_visuals_per_version_parallel
from scripts.text_associations import generate_association_per_version_parallel
from scripts.clustering import generate_cluster_visuals_per_version
from scripts.topics_modeling import generate_lda_topics_per_version
from scripts.sentiment import visualize_sentiment_per_version


def main():
    
    df = preprocess_data("data/spotify_reviews.csv")
    print(df)
    save_data(df, "data/processed_reviews.csv")
    create_version_directories(df, base_dir="results")
    generate_visuals_per_version_parallel(df)
    generate_association_per_version_parallel(df)
    generate_cluster_visuals_per_version(df, base_dir="results")
    generate_lda_topics_per_version(df, base_dir="results", n_topics=5, n_words=10)
    visualize_sentiment_per_version(df, output_dir="results")





if __name__ == "__main__":
    # clear_output_dir("results")
    # download_nltk_resources()
    # print("NLTK resources downloaded successfully.")
    # uncomment for first run
    main()
    