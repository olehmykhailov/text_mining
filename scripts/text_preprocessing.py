import re
import nltk
from nltk.corpus import stopwords


lemmatizer = nltk.WordNetLemmatizer()

stop_words = set(stopwords.words('english'))



def clean_text(text: str) -> str:
    """
    Clean the text data by removing special characters and converting to lowercase.
    """
    
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    
    text = text.lower()
    
    
    text = ' '.join(text.split())

    return text

def tokenize_text(text: str) -> list:
    """
    Tokenize the text data into words.
    """
    return nltk.word_tokenize(text)

def remove_stopwords(tokens: list) -> list:
    """
    Remove stopwords from the tokenized text.
    """
    return [word for word in tokens if word not in stop_words]

def lemmatize_text(text: str) -> str:
    """
    Lemmatize the text data using NLTK's WordNetLemmatizer.
    """
    
    tokens = tokenize_text(text)
    
    
    tokens = remove_stopwords(tokens)
    
    
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(lemmatized_tokens)


def preprocess_text(text: str) -> str:
    """
    Perform full preprocessing on the text.
    """
    
    text = clean_text(text)
    
    
    text = lemmatize_text(text)
    
    return text


