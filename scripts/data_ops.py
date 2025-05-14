from scripts.text_preprocessing import preprocess_text
import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file and return rows with 5 worst-rated app versions.
    """
    df = pd.read_csv(file_path, encoding='iso-8859-1', usecols=['appVersion', 'content', 'score'])


    return df

def save_data(df: pd.DataFrame, file_path: str) -> None:
    """
    Save the DataFrame to a CSV file.
    """
    df.to_csv(file_path, index=False, encoding='utf-8-sig')

def filter_reviews_by_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter versions with more than 50 reviews.
    """
    review_counts = df.groupby("appVersion").size().reset_index(name='count')
    filtered_stores = review_counts.query('count >= 50')
    return df[df['appVersion'].isin(filtered_stores['appVersion'])]

def fetch_five_worst_rated_versions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch the 5 worst-rated app versions based on the average score.
    """
    worst_rated = df.groupby('appVersion')['score'].mean().nsmallest(5).reset_index()
    return df[df['appVersion'].isin(worst_rated['appVersion'])]

def preprocess_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply text preprocessing to each review in the DataFrame.
    """
    df['processed_review'] = df['content'].apply(preprocess_text)
    return df


def preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess the data.
    """
    df = load_data(file_path)
    df = filter_reviews_by_count(df)
    df = fetch_five_worst_rated_versions(df)
    df = preprocess_reviews(df)
    
    return df


