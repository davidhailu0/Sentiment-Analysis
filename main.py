from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from src.model_service import load_model_components, predict_sentiment_from_text


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads the trained TF-IDF vectorizer and Logistic Regression model
    when the FastAPI application starts up.
    """
    global vectorizer, model
    print("Attempting to load model and vectorizer components on startup...")
    try:
        # Load the components using the utility function
        vectorizer, model = load_model_components()
        if vectorizer is None or model is None:
            raise RuntimeError("Model components could not be loaded.")
        print("Model and vectorizer loaded successfully for FastAPI app.")
    except Exception as e:
        print(f"Error during model loading on startup: {e}")
        # If loading fails, subsequent requests to /predict will return an error
        # Consider making this a hard failure by raising the exception
        # if the app should not start without the model.
        # raise RuntimeError(f"Failed to load model: {e}")
    yield

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="A simple API to predict movie review sentiment (positive/negative).",
    lifespan=lifespan
)

# Global variables to hold the loaded model and vectorizer
# They will be loaded once when the application starts
# Initialized to None, will be populated by load_model_components
vectorizer = None
model = None

# Pydantic model for request body validation
class ReviewInput(BaseModel):
    review_text: str


@app.get("/")
async def read_root():
    """
    Root endpoint for the API.
    """
    return {"message": "Welcome to the Sentiment Analysis API! Use POST to /predict."}

@app.post("/predict")
async def predict_sentiment_api(review_input: ReviewInput):
    """
    Predicts the sentiment of a given review text.
    Expects a JSON body with 'review_text' field.
    """
    # Ensure model components are loaded before attempting prediction
    if vectorizer is None or model is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Please ensure 'python train.py' was run successfully and the server restarted."
        )

    review_text = review_input.review_text

    if not review_text:
        raise HTTPException(status_code=400, detail="'review_text' field is required.")

    try:
        # Call the centralized prediction function
        sentiment, confidence = predict_sentiment_from_text(review_text, vectorizer, model)

        return {
            "review": review_text,
            "sentiment": sentiment.capitalize(),
            "confidence": f"{confidence:.2f}%"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

