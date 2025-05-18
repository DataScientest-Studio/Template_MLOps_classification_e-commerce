from src.features.build_features import TextPreprocessor
from src.features.build_features import ImagePreprocessor
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
from tensorflow import keras
import pandas as pd
import argparse
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
        filepath,
        imagepath
    ):
        self.tokenizer = tokenizer
        self.lstm = lstm
        self.vgg16 = vgg16
        self.best_weights = best_weights
        self.mapper = mapper
        self.filepath = filepath
        self.imagepath = imagepath

    def preprocess_image(self, image_path, target_size):
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        
        img_array = preprocess_input(img_array)
        return img_array

    def predict(self):
        X = pd.read_csv(self.filepath)[:10] #
        
        text_preprocessor = TextPreprocessor()
        image_preprocessor = ImagePreprocessor(self.imagepath)
        text_preprocessor.preprocess_text_in_df(X, columns=["description"])
        image_preprocessor.preprocess_images_in_df(X)

        sequences = self.tokenizer.texts_to_sequences(X["description"])
        padded_sequences = pad_sequences(sequences, maxlen=10, padding="post", truncating="post")

        target_size = (224, 224, 3)
        images = X["image_path"].apply(lambda x: self.preprocess_image(x, target_size))
        images = tf.convert_to_tensor(images.tolist(), dtype=tf.float32)

        lstm_proba = self.lstm.predict([padded_sequences])
        vgg16_proba = self.vgg16.predict([images])

        concatenate_proba = (
            self.best_weights[0] * lstm_proba + self.best_weights[1] * vgg16_proba
        )
        final_predictions = np.argmax(concatenate_proba, axis=1)

        return {
            i: self.mapper[str(final_predictions[i])]
            for i in range(len(final_predictions))
        }



def main():
    # Load environment variables from .env file
    load_dotenv()

    # Load paths from environment variables
    tokenizer_config_path = os.environ.get("TOKENIZER_CONFIG_PATH", "../../../models/tokenizer_config.json")
    lstm_model_path = os.environ.get("LSTM_MODEL_PATH", "../../../models/best_lstm_model.h5")
    vgg16_model_path = os.environ.get("VGG16_MODEL_PATH", "../../../models/best_vgg16_model.h5")
    best_weights_path = os.environ.get("BEST_WEIGHTS_PATH", "../../../models/best_weights.json")
    mapper_path = os.environ.get("MAPPER_PATH", "../../../models/mapper.json")
    dataset_path = os.environ.get("DATASET_PATH", "../../../data/raw/X_train_update.csv")
    images_path = os.environ.get("IMAGES_PATH", "../../../data/raw/image_train")
    predictions_path = os.environ.get("PREDICTIONS_PATH", "../../../data/predictions/predictions.json")

    parser = argparse.ArgumentParser(description= "Input data")
    
    parser.add_argument("--dataset_path", default = dataset_path, type=str,help="File path for the input CSV file.")
    parser.add_argument("--images_path", default = images_path, type=str,  help="Base path for the images.")
    args = parser.parse_args()

    
    # Charger les configurations et modèles
    with open(tokenizer_config_path, "r", encoding="utf-8") as json_file:
        tokenizer_config = json_file.read()
    tokenizer = keras.preprocessing.text.tokenizer_from_json(tokenizer_config)

    lstm = keras.models.load_model(lstm_model_path)
    vgg16 = keras.models.load_model(vgg16_model_path)

    with open(best_weights_path, "r") as json_file:
        best_weights = json.load(json_file)

    with open(mapper_path, "r") as json_file:
        mapper = json.load(json_file)
        
    
    predictor = Predict(
        tokenizer=tokenizer,
        lstm=lstm,
        vgg16=vgg16,
        best_weights=best_weights,
        mapper=mapper,
        filepath= args.dataset_path,
        imagepath = args.images_path,
    )

    # Création de l'instance Predict et exécution de la prédiction
    predictions = predictor.predict()

    # Sauvegarde des prédictions
    with open(predictions_path, "w", encoding="utf-8") as json_file:
        json.dump(predictions, json_file, indent=2)


if __name__ == "__main__":
    main()