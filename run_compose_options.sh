# Étape 1 : exécuter uniquement l'importation et le prétraitement des données, commenter le reste des commandes sur ce fichier.
docker-compose up data_service

# Étape 2 : Entraînement du modèle, cette étape requière que les étapes précédentes soient terminées. Docker-compose se chargera de démarrer toute étape nécessaire.
docker-compose up training_service

# Étape 3 : Exécution du modèle, cette étape est une étape test et requière que les étapes précédentes soient terminées. Docker-compose se chargera de démarrer toute étape nécessaire.
docker-compose up prediction_service

# Etape 4 : Partie authentificaiton
docker-compose up auth_service

# Etape 4 : Exécuter l'API de prédiction avec FastAPI
docker-compose up serving_service

# Si vous souhaitez uniquement exécuter l'API de prédiction parce que le modèle est déja entraîné et disponible, pensez aux scripts de lancement spécifiques à chaque service : /src/models/`service_name`/up.sh et /src/data/build_data/up.sh