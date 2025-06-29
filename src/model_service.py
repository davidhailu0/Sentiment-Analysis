import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os
import numpy as np

# Define paths for saving/loading model components
MODELS_DIR = 'models'
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
MODEL_PATH = os.path.join(MODELS_DIR, 'logistic_regression_model.pkl')

def train_model():
    """
    Trains a Logistic Regression model for sentiment analysis using TF-IDF.
    Saves the trained vectorizer and model to disk in the 'models' directory.
    """
    print("Starting model training...")

    # Create a small, synthetic dataset for demonstration
    # In a real project, this data would typically be loaded from a 'data/' directory.
    df = pd.read_csv('data/IMDB-Dataset.csv')

    X = df['review']
    y = df['sentiment']

    # For this small dataset, we'll train on the full set to ensure model stability
    X_train, y_train = X, y

    print(f"Dataset size: {len(X_train)} samples.")
    print("Vectorizing text using TF-IDF...")

    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)

    print("Training Logistic Regression model...")
    model = LogisticRegression(random_state=42, solver='liblinear', max_iter=100)
    model.fit(X_train_vec, y_train)

    # Create a 'models' directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save the trained vectorizer and model
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(model, MODEL_PATH)

    print(f"Vectorizer saved to: {VECTORIZER_PATH}")
    print(f"Model saved to: {MODEL_PATH}")
    print("Training complete.")

def load_model_components():
    """
    Loads the pre-trained TF-IDF vectorizer and Logistic Regression model from disk.
    Returns the vectorizer and model, or (None, None) if files are not found/loaded.
    """
    if not os.path.exists(VECTORIZER_PATH) or not os.path.exists(MODEL_PATH):
        print(f"Warning: Model or vectorizer files not found in '{MODELS_DIR}'.")
        print("Please run 'python train.py' first to train the model.")
        return None, None
    try:
        vectorizer = joblib.load(VECTORIZER_PATH)
        model = joblib.load(MODEL_PATH)
        return vectorizer, model
    except Exception as e:
        print(f"Error loading model components: {e}")
        return None, None

def predict_sentiment_from_text(review_text: str, vectorizer, model):
    """
    Predicts the sentiment of a given review text using the provided vectorizer and model.
    Args:
        review_text (str): The text of the review.
        vectorizer: The fitted TF-IDF vectorizer.
        model: The trained Logistic Regression model.
    Returns:
        tuple: (sentiment_label: str, confidence_score: float)
    """
    if vectorizer is None or model is None:
        raise ValueError("Model components are not loaded.")

    # Transform the input review text
    review_vec = vectorizer.transform([review_text])

    # Predict the sentiment
    prediction = model.predict(review_vec)[0]

    # Get the confidence score
    probabilities = model.predict_proba(review_vec)[0]
    predicted_class_index = np.where(model.classes_ == prediction)[0][0]
    confidence = probabilities[predicted_class_index] * 100

    return prediction, confidence
