import argparse
import os
from src.model_service import load_model_components, predict_sentiment_from_text

def run_cli_prediction():
    """
    Parses command-line arguments and predicts sentiment for a given review text.
    This function serves as the entry point for the 'predict.py' script.
    """
    parser = argparse.ArgumentParser(description="Predict sentiment of a movie review.")
    parser.add_argument("review_text", type=str,
                        help="The text of the movie review to analyze.")

    args = parser.parse_args()
    review_text = args.review_text

    # Load model components once
    vectorizer, model = load_model_components()

    if vectorizer is None or model is None:
        print("Error: Model components could not be loaded. Please ensure 'python train.py' has been run.")
        return

    try:
        # Predict sentiment using the centralized function
        sentiment, confidence = predict_sentiment_from_text(review_text, vectorizer, model)

        # Print the result
        print(f"Review: \"{review_text}\"")
        print(f"Sentiment: {sentiment.capitalize()}")
        print(f"Confidence: {confidence:.2f}%")
    except Exception as e:
        print(f"An error occurred during prediction: {e}")

# This part is meant to be called by the top-level predict.py script
# It's here for completeness if you were to run this file directly for testing,
# but the primary entry point will be the wrapper predict.py.
if __name__ == "__main__":
    run_cli_prediction()
