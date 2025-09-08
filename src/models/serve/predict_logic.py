import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
from tensorflow import keras
import pandas as pd
import nltk
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import math
from PIL import Image
from io import BytesIO
import pkg_resources
import os
from dotenv import load_dotenv

class Predict:
    def __init__(
        self,
        tokenizer,
        lstm,
        vgg16,
        best_weights,
        mapper,
        data,
        image_data # Preprocessed image tensor
    ):
        self.tokenizer = tokenizer
        self.lstm = lstm
        self.vgg16 = vgg16
        self.best_weights = best_weights
        self.mapper = mapper
        self.data = data
        self.image_data = image_data # Already preprocessed tensor

    def predict(self):
        
        text_preprocessor = TextPreprocessor()
        text_preprocessor.preprocess_text_in_df(self.data, columns=["description"])

        sequences = self.tokenizer.texts_to_sequences(self.data["description"])
        padded_sequences = pad_sequences(
            sequences, maxlen=10, padding="post", truncating="post"
        )

        image_tensor = tf.expand_dims(self.image_data, axis=0)  # Add batch dimension

        # Predict probabilities
        lstm_proba = self.lstm.predict([padded_sequences])
        vgg16_proba = self.vgg16.predict([image_tensor]) # Use preprocessed tensor

        # Combine probabilities
        concatenate_proba = (
            self.best_weights[0] * lstm_proba + self.best_weights[1] * vgg16_proba
        )
        final_predictions = np.argmax(concatenate_proba, axis=1)  # Predicted class indices
        confidence_scores = np.max(concatenate_proba, axis=1)  # Maximum probabilities (confidence)

        # Create a structured output with label, confidence, and prediction
        results = [
            {
                "label": self.mapper[str(final_predictions[i])],  # Map index to label
                "confidence": float(confidence_scores[i]),  # Confidence score
                "prediction": int(final_predictions[i])  # Predicted class index
            }
            for i in range(len(final_predictions))
        ]

        return results

class ImagePreprocessor: ## I wonder if it will be used 
    def __init__(self, image_data):
        self.image_data = image_data

    def preprocess_images_in_df(self, df):
        # Assuming 'image_column' contains image bytes
        df['processed_images'] = df['image_column'].apply(self.process_image)

    def process_image(self, image_bytes):
        from PIL import Image
        import io

        try:
            image = Image.open(io.BytesIO(image_bytes))
            # Example processing: resizing the image
            image = image.resize((128, 128))
            return image
        except Exception as e:
            print(f"Error processing image: {e}")
            return None

class TextPreprocessor:
    def __init__(self):
        nltk.download("punkt")
        nltk.download("stopwords")
        nltk.download("wordnet")
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(
            stopwords.words("french")
        ) 

    def preprocess_text(self, text):

        if isinstance(text, float) and math.isnan(text):
            return ""
        # Supprimer les balises HTML
        text = BeautifulSoup(text, "html.parser").get_text()

        # Supprimer les caractères non alphabétiques
        text = re.sub(r"[^a-zA-Z]", " ", text)

        # Tokenization
        words = word_tokenize(text.lower())

        # Suppression des stopwords et lemmatisation
        filtered_words = [
            self.lemmatizer.lemmatize(word)
            for word in words
            if word not in self.stop_words
        ]

        return " ".join(filtered_words[:10])

    def preprocess_text_in_df(self, df, columns):
        for column in columns:
            df[column] = df[column].apply(self.preprocess_text)

def preprocess_image(image_input, target_size=(224, 224)):
        """Wrapper function to preprocess image."""
        if isinstance(image_input, bytes):  # Handle raw bytes
            image = Image.open(BytesIO(image_input)).convert("RGB")
            image = image.resize(target_size)
        else:  # Handle file paths or file-like objects
            image = load_img(image_input, target_size=target_size)
        
        return  tf.convert_to_tensor(img_to_array(image), dtype=tf.float32) # Convert the image to a NumPy array and then to a TensorFlow tensor

def make_prediction(data_api, image_data, tokenizer, lstm, vgg16, best_weights, mapper):
    
     # Predictor instance
    predictor = Predict(
        tokenizer=tokenizer,
        lstm=lstm,
        vgg16=vgg16,
        best_weights=best_weights,
        mapper=mapper,
        data=data_api, # data_api is the dataframe from the API
        image_data=image_data # image_data is the image bytes from the API
    )

    # Make predictions
    predictions = predictor.predict()

    return predictions

# Preprocessing function for images : moved to predict_logic.py
#def preprocess_image(image_bytes, target_size=(224, 224, 3)):
#    image = Image.open(BytesIO(image_bytes)).convert("RGB")
#    image = image.resize(target_size)
#    image_array = np.array(image) / 255.0  # Normalize pixel values
#    return image_array

# Another version of preprocess_image
#   img = load_img(image_data, target_size=target_size)
     #   img_array = img_to_array(img)
      #  
     #   img_array = preprocess_input(img_array)
     #   return img_array