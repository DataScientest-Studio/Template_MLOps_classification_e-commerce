import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

class TextLSTMModel:
    def __init__(self, max_words=10000, max_sequence_length=10):
        self.max_words = max_words
        self.max_sequence_length = max_sequence_length
        self.tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
        self.model = None

    def preprocess_and_fit(self, X_train, y_train, X_val, y_val):
        self.tokenizer.fit_on_texts(X_train['description'])

        tokenizer_config = self.tokenizer.to_json()
        with open('tokenizer_config.json', 'w', encoding='utf-8') as json_file:
          json_file.write(tokenizer_config)

        train_sequences = self.tokenizer.texts_to_sequences(X_train['description'])
        train_padded_sequences = pad_sequences(train_sequences, maxlen=self.max_sequence_length, padding='post', truncating='post')

        val_sequences = self.tokenizer.texts_to_sequences(X_val['description'])
        val_padded_sequences = pad_sequences(val_sequences, maxlen=self.max_sequence_length, padding='post', truncating='post')

        text_input = Input(shape=(self.max_sequence_length,))
        embedding_layer = Embedding(input_dim=self.max_words, output_dim=128)(text_input)
        lstm_layer = LSTM(128)(embedding_layer)
        output = Dense(27, activation='softmax')(lstm_layer)

        self.model = Model(inputs=[text_input], outputs=output)

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        lstm_callbacks = [ModelCheckpoint(filepath='best_lstm_model.h5', save_best_only=True),  # Enregistre le meilleur modèle
        EarlyStopping(patience=3, restore_best_weights=True),  # Arrête l'entraînement si la performance ne s'améliore pas
        TensorBoard(log_dir='./logs')  # Enregistre les journaux pour TensorBoard
        ]

        self.model.fit(
            [train_padded_sequences],
            tf.keras.utils.to_categorical(y_train, num_classes=27),
            epochs=100,
            batch_size=32,
            validation_data=([val_padded_sequences], tf.keras.utils.to_categorical(y_val, num_classes=27)),
            callbacks=lstm_callbacks
        )
