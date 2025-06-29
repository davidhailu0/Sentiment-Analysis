# Sentiment Analysis Mini-Project

This project demonstrates a basic machine learning pipeline for text classification, specifically sentiment analysis (positive vs. negative). It uses scikit-learn for TF-IDF vectorization and Logistic Regression for classification. The project is structured to follow best practices for modularity and maintainability.

## Project Structure

```
.
├── src/
│ ├── init.py # Makes src a Python package
│ ├── model_service.py # Contains functions for model training, loading, and prediction logic
│ └── cli_predict.py # Contains logic for the command-line prediction script
├── models/ # Stores trained vectorizer and model files (.pkl)
├── data/ # (Optional) Placeholder for raw or processed datasets
├── requirements.txt
├── main.py # The main FastAPI application entry point
├── train.py # Script to initiate model training
├── predict.py # Script to run command-line predictions
└── README.md

```

## Setup

### 1. Create Directories

First, create the necessary directories:

```bash
mkdir src
mkdir models
```

### 2. Create Files

Place the code provided in the respective files:

requirements.txt (same as before)

main.py (updated from fastapi_app_py)

train.py (updated)

predict.py (updated)

src/**init**.py (empty file)

src/model_service.py (new)

src/cli_predict.py (new)

README.md (this file)

### 3. Install Dependencies

It's recommended to use a virtual environment.

#### Create a virtual environment (optional but recommended)

python -m venv venv

#### Activate the virtual environment

# On Windows:

#### venv\Scripts\activate

# On macOS/Linux:

#### source venv/bin/activate

# Install the required packages

pip install -r requirements.txt

# For FastAPI, you'll also need uvicorn

pip install uvicorn

Usage

### 1. Train the Model

Before you can make predictions or run the API, you need to train the machine learning model. This script will use a small embedded dataset, then save the trained TF-IDF vectorizer and Logistic Regression model into the models/ directory.

```
python train.py
```

You should see output indicating the training process and where the model files are saved.

### 2. Run Command-Line Predictions

Once the model is trained, you can use the predict.py script to get sentiment predictions for new review texts from your terminal.

```
python predict.py "I loved this movie! It was fantastic and captivating."
```

Example Output:

Review: "I loved this movie! It was fantastic and captivating."
Sentiment: Positive
Confidence: 90.00%

If the model files are not found, the script will prompt you to run train.py first.

### 3. Run the FastAPI Application

To expose your sentiment analysis model as a web API, run the FastAPI application:

```
uvicorn main:app --reload
```

The API will typically be available at http://127.0.0.1:8000.

You can access the interactive API documentation (Swagger UI) at http://127.0.0.1:8000/docs. This allows you to test the /predict endpoint directly in your browser.

To test the endpoint using curl:

```
curl -X POST -H "Content-Type: application/json" \
 -d '{"review_text": "This movie was absolutely captivating!"}' \
 [http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)
```

Expected JSON Output:

```
{
"confidence": "90.00%",
"review": "This movie was absolutely captivating!",
"sentiment": "Positive"
}
```

This structured approach makes the project easier to understand, test, and scale in the future.
