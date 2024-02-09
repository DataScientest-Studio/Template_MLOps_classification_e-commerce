from features.build_features import TextPreprocessor
from features.build_features import ImagePreprocessor
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
from tensorflow import keras
import pandas as pd


class Predict:
    def __init__(
        self, tokenizer, lstm, vgg16, best_weights, mapper, filepath="../data/external/"
    ):
        self.tokenizer = tokenizer
        self.lstm = lstm
        self.vgg16 = vgg16
        self.best_weights = best_weights
        self.filepath = filepath
        self.mapper = mapper

    def preprocess_image(self, image_path, target_size):
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        return img_array

    def predict(self, X):
        text_preprocessor = TextPreprocessor()
        image_preprocessor = ImagePreprocessor(self.filepath)
        text_preprocessor.preprocess_text_in_df(X, columns=["description"])
        image_preprocessor.preprocess_images_in_df(X)

        sequences = self.tokenizer.texts_to_sequences(X["description"])
        padded_sequences = pad_sequences(
            sequences, maxlen=10, padding="post", truncating="post"
        )

        target_size = (224, 224, 3)
        images = X["image_path"].apply(lambda x: self.preprocess_image(x, target_size))
        images = tf.convert_to_tensor(images.tolist(), dtype=tf.float32)

        lstm_proba = self.lstm.predict([padded_sequences])
        vgg16_proba = self.vgg16.predict([images])

        concatenate_proba = (
            self.best_weights[0] * lstm_proba + self.best_weights[1] * vgg16_proba
        )
        final_predictions = np.argmax(concatenate_proba, axis=1)

        return f"Catégorie : {self.mapper.loc[final_predictions].values[0]}"


with open("../models/tokenizer_config.json", "r", encoding="utf-8") as json_file:
    tokenizer_config = json_file.read()
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_config)
lstm = keras.models.load_model("../models/best_lstm_model.h5")
vgg16 = keras.models.load_model("../models/best_vgg16_model.h5")

with open("../models/best_weights.json", "r") as json_file:
    best_weights = json.load(json_file)

with open("../models/mapper.json", "r") as json_file:
    mapper = json.load(json_file)

X = pd.read_csv("../data/external/test.csv")
predictor = Predict(
    tokenizer=tokenizer,
    lstm=lstm,
    vgg16=vgg16,
    best_weights=best_weights,
    mapper=mapper,
)

with open("../data/external/predictions.json", "w", encoding="utf-8") as json_file:
    json_file.write(predictor.predict(X))
