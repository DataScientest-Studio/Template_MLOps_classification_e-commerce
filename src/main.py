from preprocessing.text_preprocessing import TextPreprocessor
from preprocessing.image_preprocessing import ImagePreprocessor
from models.text_lstm import TextLSTMModel
from models.image_vgg16 import ImageVGG16Model
from data.import_data import DataImporter
from models.Concatenate import concatenate
import tensorflow as tf
from tensorflow import keras
from models.Predict import Predict
import json


data_importer = DataImporter('data/data')
df = data_importer.load_data()
X_train, X_val, _, y_train, y_val, _ = data_importer.split_train_test(df)

# Preprocess text and images
text_preprocessor = TextPreprocessor()
image_preprocessor = ImagePreprocessor('data/data')
text_preprocessor.preprocess_text_in_df(X_train, columns=['description'])
text_preprocessor.preprocess_text_in_df(X_val, columns=['description'])
image_preprocessor.preprocess_images_in_df(X_train)
image_preprocessor.preprocess_images_in_df(X_val)

# Train LSTM model
text_lstm_model = TextLSTMModel()
text_lstm_model.preprocess_and_fit(X_train, y_train, X_val, y_val)

# Train VGG16 model
image_vgg16_model = ImageVGG16Model()
image_vgg16_model.preprocess_and_fit(X_train, y_train, X_val, y_val)

with open('tokenizer_config.json', 'r', encoding='utf-8') as json_file:
    tokenizer_config = json_file.read()
tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(
    tokenizer_config
)
lstm = keras.models.load_model('best_lstm_model.h5')
vgg16 = keras.models.load_model('best_vgg16_model.h5')

model_concatenate = concatenate(tokenizer, lstm, vgg16)
lstm_proba, vgg16_proba, new_y_train = model_concatenate.predict(X_train, y_train)
best_weights = model_concatenate.optimize(lstm_proba, vgg16_proba, new_y_train)

num_classes = 27

proba_lstm = keras.layers.Input(shape=(num_classes,)) 
proba_vgg16 = keras.layers.Input(shape=(num_classes,))

weighted_proba = keras.layers.Lambda(lambda x: best_weights[0] * x[0] + best_weights[1] * x[1])([proba_lstm, proba_vgg16])

concatenate_model = keras.models.Model(inputs=[proba_lstm, proba_vgg16], outputs=weighted_proba)

# Enregistrer le modèle au format h5
concatenate_model.save('concatenate.h5')

with open('tokenizer_config.json', 'r', encoding='utf-8') as json_file:
    tokenizer_config = json_file.read()
tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(
    tokenizer_config
)
lstm = keras.models.load_model('best_lstm_model.h5')
vgg16 = keras.models.load_model('best_vgg16_model.h5')

with open('best_weights.json', 'r') as json_file:
    best_weights = json.load(json_file)

predictor = Predict(tokenizer, lstm, vgg16, best_weights, filepath='data/data')
final_predictions = predictor.predict(X_val, y_val)
