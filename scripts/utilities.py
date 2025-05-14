import nltk
import re
import os
import shutil

def download_nltk_resources():
    """
    Download the necessary NLTK resources.
    """
    try:
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')
    except Exception as e:
        raise RuntimeError(f"Failed to download NLTK resources: {e}")




def clear_output_dir(output_dir: str):
    """
    Clear the output directory.
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

def create_version_directories(df, base_dir, subfolders=None):
    """
    Create directories for each app version based on the 'appVersion' column in the DataFrame.
    """
    if subfolders is None:
        subfolders = ["graphs", "text_visuals", "clusterisation", "topics", "sentiment"]

    os.makedirs(base_dir, exist_ok=True)

    for version in df['appVersion'].unique():

        version_dir = os.path.join(base_dir, version)
        os.makedirs(version_dir, exist_ok=True)

        for sub in subfolders:
            sub_dir = os.path.join(version_dir, sub)
            os.makedirs(sub_dir, exist_ok=True)

