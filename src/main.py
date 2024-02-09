from features.build_features import DataImporter, TextPreprocessor, ImagePreprocessor
from models.train_model import TextLSTMModel, ImageVGG16Model, concatenate
from tensorflow import keras
import pickle
import tensorflow as tf


data_importer = DataImporter()
df = data_importer.load_data()
X_train, X_val, _, y_train, y_val, _ = data_importer.split_train_test(df)

# Preprocess text and images
text_preprocessor = TextPreprocessor()
image_preprocessor = ImagePreprocessor()
text_preprocessor.preprocess_text_in_df(X_train, columns=["description"])
text_preprocessor.preprocess_text_in_df(X_val, columns=["description"])
image_preprocessor.preprocess_images_in_df(X_train)
image_preprocessor.preprocess_images_in_df(X_val)

# Train LSTM model
print("Training LSTM Model")
text_lstm_model = TextLSTMModel()
text_lstm_model.preprocess_and_fit(X_train, y_train, X_val, y_val)
print("Finished training LSTM")

print("Training VGG")
# Train VGG16 model
image_vgg16_model = ImageVGG16Model()
image_vgg16_model.preprocess_and_fit(X_train, y_train, X_val, y_val)
print("Finished training VGG")

with open("models/tokenizer_config.json", "r", encoding="utf-8") as json_file:
    tokenizer_config = json_file.read()
tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_config)
lstm = keras.models.load_model("models/best_lstm_model.h5")
vgg16 = keras.models.load_model("models/best_vgg16_model.h5")

print("Training the concatenate model")
model_concatenate = concatenate(tokenizer, lstm, vgg16)
lstm_proba, vgg16_proba, new_y_train = model_concatenate.predict(X_train, y_train)
best_weights = model_concatenate.optimize(lstm_proba, vgg16_proba, new_y_train)
print("Finished training concatenate model")

with open("models/best_weights.pkl", "wb") as file:
    pickle.dump(best_weights, file)

num_classes = 27

proba_lstm = keras.layers.Input(shape=(num_classes,))
proba_vgg16 = keras.layers.Input(shape=(num_classes,))

weighted_proba = keras.layers.Lambda(
    lambda x: best_weights[0] * x[0] + best_weights[1] * x[1]
)([proba_lstm, proba_vgg16])

concatenate_model = keras.models.Model(
    inputs=[proba_lstm, proba_vgg16], outputs=weighted_proba
)

# Enregistrer le modèle au format h5
concatenate_model.save("models/concatenate.h5")
