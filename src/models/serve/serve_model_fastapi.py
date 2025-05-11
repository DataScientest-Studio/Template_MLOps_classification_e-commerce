from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from typing import Optional
from PIL import Image
import numpy as np
import pandas as pd
import io
import tensorflow as tf
from io import BytesIO
from src.models.serve.predict_logic import make_prediction, preprocess_image


app = FastAPI()

class SubPrediction(BaseModel):
    label: str
    confidence: float
    prediction: int

class PredictionResponse(BaseModel):
    predictions: list[SubPrediction] # prediction result
    message: str # response message

@app.get("/status")
def status():
    return {"message": "Model Serving via FastAPI is running !"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_code(
    product_identifier: str = Form(...), # An integer ID for the product. This ID is used to associate the product with its corresponding product type code.
    designation: str = Form(...), # The product title, a short text summarizing the product.
    description: str = Form(...),  # A more detailed text describing the product. Not all the merchants use this field, so to retain originality of the data, the description field can contain NaN value for many products.
    product_id: str = Form (...), # An unique ID for the product.
    imageid: str = Form (...), # An unique ID for the image associated with the product.
    image: UploadFile = File(...)  # Receive image as file)
):
    
    # Process the image input as bytes
    image_data = await image.read()
    processed_image = preprocess_image(image_data)

    # Convert to TensorFlow tensor (optional, for model input)
    #image_tensor = tf.convert_to_tensor([processed_image], dtype=tf.float32)
    
    # construct a df with textual data including description
    df = pd.DataFrame({"product_identifier":[product_identifier], "description": [description], "designation": [designation], "product_id": [product_id], "imageid": [imageid]})

    predictions = make_prediction(df, processed_image)

    # Convert the list of dictionaries to a list of SubPrediction objects
    sub_predictions = [SubPrediction(**prediction) for prediction in predictions]

    # endpoint return value : {"prediction": predictions["prediction"], "code": predictions["label"], "confidence": predictions["confidence"], "message": "Prediction successful"}
    return {
        "predictions": sub_predictions,
        "message": "Prediction successful"
    }