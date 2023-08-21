from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import pandas as pd

class ImageVGG16Model:
    def __init__(self):
        self.model = None

    def preprocess_and_fit(self, X_train, y_train, X_val, y_val):
        # Paramètres
        batch_size = 32
        num_classes = 27

        df_train = pd.concat([X_train, y_train.astype(str)], axis=1)
        df_val = pd.concat([X_val, y_val.astype(str)], axis=1)

        # Créer un générateur d'images pour le set d'entraînement
        train_datagen = ImageDataGenerator()  # Normalisation des valeurs de pixel
        train_generator = train_datagen.flow_from_dataframe(
            dataframe=df_train,
            x_col='image_path',
            y_col='prdtypecode',
            target_size=(224, 224),  # Adapter à la taille d'entrée de VGG16
            batch_size=batch_size,
            class_mode='categorical',  # Utilisez 'categorical' pour les entiers encodés en one-hot
            shuffle=True
        )

        # Créer un générateur d'images pour le set de validation
        val_datagen = ImageDataGenerator()  # Normalisation des valeurs de pixel
        val_generator = val_datagen.flow_from_dataframe(
            dataframe=df_val,
            x_col='image_path',
            y_col='prdtypecode',
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False  # Pas de mélange pour le set de validation
        )

        image_input = Input(shape=(224, 224, 3))  # Adjust input shape according to your images

        vgg16_base = VGG16(include_top=False, weights='imagenet', input_tensor=image_input)

        x = vgg16_base.output
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)  # Add some additional layers if needed
        output = Dense(num_classes, activation='softmax')(x)

        self.model = Model(inputs=vgg16_base.input, outputs=output)

        for layer in vgg16_base.layers:
            layer.trainable = False

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        vgg_callbacks = [ModelCheckpoint(filepath='best_vgg16_model.h5', save_best_only=True),  # Enregistre le meilleur modèle
        EarlyStopping(patience=3, restore_best_weights=True),  # Arrête l'entraînement si la performance ne s'améliore pas
        TensorBoard(log_dir='./logs')  # Enregistre les journaux pour TensorBoard
        ]

        self.model.fit(
            train_generator,
            epochs=100,
            validation_data=val_generator,
            callbacks=vgg_callbacks
        )
