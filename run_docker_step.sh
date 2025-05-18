# Étape 1 : Importation des données
docker-compose up import-data

# Étape 2 : Prétraitement (copie de /data/raw vers /data/preprocessed)
docker-compose up files-preprocess

# Étape 3 : Entraînement
docker-compose up train