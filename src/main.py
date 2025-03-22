# Importation des bibliothèques nécessaires
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.applications import VGG16  # Importation correcte de VGG16
from keras.utils import to_categorical
import numpy as np

# Supposons que tu aies tes données d'entraînement et de validation
# Remplace ces données par tes propres données

# Exemple de création de données fictives (à remplacer par tes propres données)
X_train = np.random.random((100, 224, 224, 3))  # 100 images de taille 224x224x3
X_val = np.random.random((20, 224, 224, 3))  # 20 images de taille 224x224x3

# Exemple de labels (à remplacer par tes propres labels)
y_train = np.random.randint(0, 11, 100)  # 100 labels avec des valeurs entre 0 et 10 (11 classes)
y_val = np.random.randint(0, 11, 20)  # 20 labels avec des valeurs entre 0 et 10 (11 classes)

# Encoder les labels en one-hot avec 11 classes
y_train = to_categorical(y_train, num_classes=11)
y_val = to_categorical(y_val, num_classes=11)

# Définir le modèle
model = Sequential()

# Charger le modèle VGG16 pré-entraîné, sans les couches de classification finales
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.add(base_model)

# Ajouter des couches supplémentaires
model.add(Flatten())  # Aplatir les sorties du modèle VGG16
model.add(Dense(256, activation='relu'))  # Couches denses supplémentaires
model.add(Dense(11, activation='softmax'))  # 11 classes pour la classification

# Compiler le modèle avec l'optimiseur 'adam' et la fonction de perte 'categorical_crossentropy'
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Afficher un résumé du modèle pour vérifier sa structure
model.summary()

# Entraîner le modèle avec les données d'entraînement et de validation
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Sauvegarder le modèle entraîné si nécessaire
# model.save('vgg16_model.h5')
