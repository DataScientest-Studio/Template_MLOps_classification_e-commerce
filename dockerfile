# Utilisation d'une image Python optimisée
FROM python:3.9-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers nécessaires
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste du code source
COPY src/ src/
COPY models/ models/
COPY data/preprocessed/ data/preprocessed/

# Définir la commande par défaut pour exécuter le modèle
CMD ["python", "src/predict.py"]
