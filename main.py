import json
import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import pickle

# Initialisation de l'API FastAPI
api = FastAPI()

# Charger le modèle LSTM
model_path = "models/best_lstm_model.h5"
model = tf.keras.models.load_model(model_path)

# Charger le tokenizer
tokenizer_path = "models/tokenizer_config.json" 

with open(tokenizer_path, "r") as file:
    tokenizer_json = file.read()  # lire le fichier json
tokenizer = tokenizer_from_json(tokenizer_json)  

# Charger le mapper des classes
mapper_path = "models/mapper.pkl"
with open(mapper_path, "rb") as file:
    class_mapper = pickle.load(file)  # Dictionnaire {id: label}

# Définition du format de la requête
class InputText(BaseModel):
    text: str

# Endpoint de test
@api.get("/status")
def get_status():
    return {"status": "L'API fonctionne correctement"}

# Endpoint de prédiction
@api.post("/predict")
def predict(data: InputText):
    # Tokenization et padding du texte

    sequence = tokenizer.texts_to_sequences([data.text])
    padded_sequence = pad_sequences(sequence, maxlen=10) 

    # Prédiction
    prediction = model.predict(padded_sequence)
    predicted_class = np.argmax(prediction)

    # Mapper l'ID de la classe vers le label réel
    class_label = class_mapper.get(predicted_class, "Classe inconnue")

    return {
        "text": data.text,
        "predicted_class": int(predicted_class),
        "class_label": class_label
    }

# Endpoint racine
@api.get("/")
def lire_racine():
    return {"message": "Bienvenue sur l'API MLOps avec FastAPI et LSTM!"}

