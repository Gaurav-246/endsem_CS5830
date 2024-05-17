from fastapi import FastAPI
from tensorflow.keras.models import Sequential
import tensorflow as tf
import numpy as np
import os
from fastapi.requests import Request


app = FastAPI()

model_path = str(os.getenv("MODEL_PATH"))  # Model path from terminal
# Run "export MODEL_PATH=/path/to/model" in terminal

def load_model_weights(model_path: str):
    model = tf.keras.models.load_model(model_path)
    return model

def predict_rating(model: Sequential, data_point: np.array):
    data_point = np.expand_dims(data_point, axis=0)
    pred = model.predict(data_point)
    return str(np.argmax(pred)+1)      # Returns predicted sentiment rating

def read_review(review : str):
    return np.random.rand(100)

@app.post("/read_review")
async def predict_senti(request : Request):
    review_text = await request.body()
    review_text = review_text.decode('utf-8')   # Decode the bytes to str
    processed_text = read_review(review_text)   # Pre-process it
    model = load_model_weights(model_path)      
    rating = predict_rating(model, processed_text)  # predict the score
    return {"Sentiment score : ": rating}