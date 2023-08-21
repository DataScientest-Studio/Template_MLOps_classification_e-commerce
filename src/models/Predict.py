from preprocessing.text_preprocessing import TextPreprocessor
from preprocessing.image_preprocessing import ImagePreprocessor
from models.Concatenate import Concatenate
from sklearn.metrics import f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

class Predict:
  def __init__(self, tokenizer, lstm, vgg16, best_weights, filepath='data/external'):
    self.tokenizer = tokenizer
    self.lstm = lstm
    self.vgg16 = vgg16
    self.best_weights = best_weights
    self.filepath = filepath
  
  def preprocess_image(self, image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    return img_array

  def predict(self, X, y=None):
    text_preprocessor = TextPreprocessor()
    image_preprocessor = ImagePreprocessor(self.filepath)
    text_preprocessor.preprocess_text_in_df(X, columns=['description'])
    image_preprocessor.preprocess_images_in_df(X)
    
    sequences = self.tokenizer.texts_to_sequences(X['description'])
    padded_sequences = pad_sequences(sequences, maxlen=10, padding='post', truncating='post')

    target_size = (224, 224, 3) 
    images = X['image_path'].apply(lambda x: self.preprocess_image(x, target_size))
    images = tf.convert_to_tensor(images.tolist(), dtype=tf.float32)

    lstm_proba = self.lstm.predict([padded_sequences])
    vgg16_proba = self.vgg16.predict([images])

    concatenate_proba = self.best_weights[0]*lstm_proba + self.best_weights[1]*vgg16_proba
    final_predictions = np.argmax(concatenate_proba, axis=1)

    if y is not None:
      weighted_f1 = f1_score(y, final_predictions, average='weighted')

      plt.figure(figsize=(12, 8))
      sns.heatmap(pd.crosstab(y, final_predictions), annot=True)
      plt.savefig('matrice_confusion.png', dpi=300)

    return final_predictions