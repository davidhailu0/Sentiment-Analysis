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

#### 1. Download the Dataset

Download the IMDB dataset from [here](https://storage.googleapis.com/kaggle-data-sets/134715/320111/compressed/IMDB%20Dataset.csv.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250629%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250629T173337Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=3366d3e625d9f64152bad7e35d1584104a3adabb043c3e9c2f80288407e3266d609147daaed63b777eb51be55aaddc105762b93ad236a39c08dd4e04dd10d32a0a80b46a1f0999b4e88a5effa5cd76fd2c2509aec2b428c9da31afb040492635da32c810f1d2271980a672ed2d6bf0084967d2ba24ad8a3ae8abd5a0b7f85301234b6b9a0595233c399da97563a3a3d426a0a8dbcd6cca0e045e94a0e2a130e61930e19a8560e4dfc4ee691aa87ac2b9be62c2e499a29f41a4ce36709f77e465d44a3a2ad4c7949cabde0754bb9bff7517150ed6cdcd7748457802b9bd6cfcde9b2dc13c7a3ebd266b9b1105a390e9d9725d865d65661ea02b3fc47bdbdbb2ea). Unzip the file and place the unzipped folder in the data/ directory.

#### 2. Train the Model

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
