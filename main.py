import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re
from tokenizer import makeTokens  # Ensure this tokenizer is reliable and processes URLs correctly

# Read the data from CSV
try:
    url_data = pd.read_csv("datasets/urlbadgood.csv", header=None, names=["URL", "Label"])
except Exception as e:
    raise FileNotFoundError(f"Error reading URL data file: {str(e)}")

# Map labels to integers: 'good' -> 1, 'bad' -> 0
label_mapping = {'good': 1, 'bad': 0}
url_data['Label'] = url_data['Label'].map(label_mapping)

# Ensure there are no NaN values in the URLs and labels
url_data = url_data.dropna(subset=["URL", "Label"])

# Filter out any rows where URL is empty
url_data = url_data[url_data['URL'].str.strip() != '']

# Shuffle the combined data
url_data = url_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Split data into features and labels
X_train = url_data['URL']
y_train = url_data['Label'].astype(int)  # Now this will be correctly formatted as integers

# Initialize TfidfVectorizer for better text representation
vectorizer = TfidfVectorizer(tokenizer=makeTokens, token_pattern=None)

# Transform the features using TfidfVectorizer
X_train_transformed = vectorizer.fit_transform(X_train)

# Initialize Logistic Regression model
logit = LogisticRegression()

# Train the model
logit.fit(X_train_transformed, y_train)

# Define input data model
class InputData(BaseModel):
    url: str

# Create FastAPI instance
app = FastAPI()

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify allowed origins
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Prediction endpoint
@app.post("/predict")
async def predict(input_data: InputData):
    try:
        # Preprocess the input URL
        url_processed = re.sub(r'^((?!www\.)[a-zA-Z0-9-]+\.[a-z]+)$', r'www.\1', input_data.url)
        
        # Vectorize the URL
        X_predict = vectorizer.transform([url_processed])
        
        # Make prediction
        prediction = logit.predict(X_predict)[0]  # Get the single prediction result
        
        # Map the prediction label (0 -> bad, 1 -> good)
        label_mapping = {0: "bad", 1: "good"}
        prediction_label = label_mapping[prediction]

        # Return the prediction result
        return {"url": input_data.url, "prediction": prediction_label}
    
    except Exception as e:
        # Log the error and raise an HTTP exception
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")
