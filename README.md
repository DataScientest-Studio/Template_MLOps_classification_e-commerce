# Phase 1
 
- **Définir les objectifs et métriques du projet**
- **Mettre en place l'environnement de développement avec Docker ou de simples environnements virtuels**
- **Créer les premiers pipelines de données et d'entraînement**
- **Mise en place Git pour le code et premiers tests automatisés**
- **Créer une API basique pour le modèle**
## Cadrage du Projet MLOps
Cataloguer les produits selon des données différentes (textes et images) est important pour les e-commerces puisque cela permet de réaliser des applications diverses telles que la recommandation de produits et la recherche personnalisée. Il s’agit alors de prédire le code type des produits à partir de données textuelles (désignation et description des produits) ainsi que des données d'images (image du produit).
Le client est le site internet de Rakuten, et plus particulièrement les administrateurs de ce site.
### Étapes pour cadrer le projet MLOps (Rakuten)
1. **Objectifs et Problématique**
   1. Pourquoi mettre en place un pipeline MLOps ? (automatisation, scalabilité, fiabilité…)
   2. Quels sont les objectifs ?
      1. Construire un modèle de classification d’images pour catégoriser des produits e-commerce.
      2. Déploiement du modèle deep learning en production.
   3. Quels sont les KPIs pour mesurer la performance du modèle et du pipeline MLOps ? (ex. : précision, latence, coût d’inférence…)
      4. Métriques :
         5. Accuracy : Évaluer la proportion de bonnes prédictions 
         6. F1-score : Prendre en compte le déséquilibre des classes 
         7. Temps d’inférence : Mesurer la rapidité des prédictions du modèle
2. **Données et Infrastructure**
   1. **Sources des données** : stockés dans HuggingFace: https://huggingface.co/datasets/ndiguesene/ml-datasets-image-rakuten-ecommerce/resolve/main/
      - Données textuelles (~60 MB)
      - Données images (~2.5 GB)
   2. **Stockage** : fichiers plats (CSV)
   3. **Nettoyage & Préparation** : Stratégie de gestion des données manquantes et transformation.
   4. **Infrastructure** : À définir.
3. **Développement du Modèle ML**
   1. **Choix des algorithmes** : Deep learning
   2. **Frameworks utilisés** : TensorFlow, Scikit-learn
4. **Industrialisation avec MLOps**
   1. **CI/CD pour le ML** : GitHub, GitHub Actions
   2. **Gestion des modèles** : MLflow
   3. **Orchestration** : Airflow
   4. **Monitoring & Observabilité** : Prometheus, Grafana
5. **Déploiement et Scalabilité**
   1. **Mode de déploiement** : Batch (REST API)
   2. **Infrastructure de déploiement** : Docker, Kubernetes (à confirmer)
   3. **Gestion du drift** : Retrain automatique, A/B testing, monitoring des performances
6. **Sécurité et Gouvernance**
   1. Sécurisé les APIs exposées
---
## Mettre en place l'environnement de développement reproductible
### Création de l'environnement 
#### - Sans Docker
Créez l'environnement virtuel avec les étapes suivantes sur Linux :
 
`python -m venv rakuten-project-mlops`
 
Cela va créer un dossier `rakuten-project-mlops/` contenant l’environnement virtuel.
 
### Installer les dépendances
 
Installez les dépendances nécessaires :
 
`pip install -r requirements.txt`
 
#### - Avec Docker
- Créer un Dockerfile et un `docker-compose.yml`
- Construire l’image avec `docker build -t ml_env .`
- Lancer le conteneur avec `docker-compose up`
### Importer les données dans `raw/data`
 
Ensuite, exécutez le script pour importer les données depuis le repo HuggingFace :
`python3 src/data/import_raw_data.py`
 
Arborescence des données
```
    data/raw/
    │── image_train/
    │   ├── image_123456789_product_987654321.jpg
    │   ├── image_987654321_product_123456789.jpg
    │   └── ...
    │── image_test/
    │   ├── image_111111111_product_222222222.jpg
    │   ├── image_222222222_product_111111111.jpg
    │   └── ...
```
---
## Copiez les données brutes dans `data/preprocessed/` :
 
`python3 src/data/make_dataset.py data/raw data/preprocessed`
 
## Entraînez les modèles sur l’ensemble de données et enregistrez-les dans le dossier models
 
`python3 python src/main.py`
## Démarrer les pratiques de versionning et des tests automatisés
### Suivi et gestion des versions des données et des modèles de données
1. **Initialiser DVC dans le projet**
```bash
  dvc init
```
2. **Ajouter les fichiers volumineux avec DVC**
```bash
    dvc add data/raw/image_train
    dvc add data/raw/image_test
    dvc add data/preprocessed/X_train_update.csv
    dvc add data/preprocessed/X_test_update.csv
```
3. **Créer un fichier `.gitignore` pour ignorer les fichiers volumineux suivis par DVC**
```bash
  echo "*.dvc" > .gitignore
```
### Configurer un stockage distant avec DVC
On utilisera **S3 d’AWS** pour sauvegarder les données. Pour cela, il faut d'abord installer S3 à l’aide de la commande suivante :

 
`pip install "dvc[S3]"`
 
 
Une fois S3 installé, il faudra mettre à jour le fichier de configuration `.dvc/config` pour ajouter DagsHub comme notre stockage distant.

## Partie 1 : serving, dockerisation et tests unitaires
### Restructuration des répertoires et fichiers

#### train
Ici, nous entraînons le modèle (contient les fichiers main.py et train_model.py). Le répertoire contiendra un Dockerfile spécifique pour lancer l'entraînement du modèle à la demande. Les résultats seront enregistrés dans `models` à la racine
#### predict 
Ici, on applique le modèle au données d'entraînement (contient le fichier predict.py). Le répertoire contiendra un Dockerfile spécifique pour lancer le test du modèle sur les données d'entraînement; Les résultats seront enregistrés sur data/predictions à la racine
#### evaluate 
Ici, on applique le modèle au données de test (devra être créé mais peut s'inspirer du fichier predict.py). Il contiendra un Dockerfile spécifique pour lancer le test du modèle sur les données de tes; Les résultats seront enregistrés sur data/predictions à la racine. 
##### Il semble que l'on ne puisse pas évaluer car il n'y a pas de données labélisées sur l'ensemble de test.
#### serve
Ici, on utilisera dans un premier temps requests ou FastAPI pour créer une API qui va interroger le modèle avec des données préalablement renseignées (fichier serve.py à créer) et retournera des résultats que l'on pourra mettre dans data/predictions à la racine avec un nom de fichier spécifique
Niveaux de serving : 
1. Usage de FastAPI pour créer un contneur basique de serving
2. Usage de BentoML pour automatiser le déploiement, serving etc
#### Variables d'environnement : vous devez rajouter un fichier .env sur la racine du projet
Les chemins des modèles (poids, mapper etc) ne sont plus codés en dur. Les modifications suivantes ont été apportées : 
1. ajout fichier `.env` contenant les variables d'environnement (**ce fichier ne doit pas être poussé sur le git, il peut renfermer des informations sensibles tels clés d'api etc**)
2. ajout fichier `.env.example` sur le git : fichier d'exemple montrant comment renseigner le fichier .env ( quelles sont les variables d'environnement etc)

## Conteneurisation
Docker est utilisé pour conteneuriser les différents services de l'application. Chaque service dispose d'un répertoire spécifique sur les sources.

### Variables d'environnement
Le fichier `.env` à la racine du projet, doit être mis en jour avec les répertoires de volume créés dans le conteneur (Exemple : `TOKENIZER_CONFIG_PATH=/app/models/tokenizer_config.json`).

### Services applicatifs
1. Service de données
Ici, le service effectue le téléchargement des données depuis le dépôt HuggingFace : `https://huggingface.co/datasets/ndiguesene/ml-datasets-image-rakuten-ecommerce/resolve/main/`.

`Package` : src/data/build_data/

2. Service d'entraînement

`Package` : src/models/train
Contient les fichiers : 
- .dockerignore : fichier précisant les fichiers et répertoires à ignorer lors de la construction de l'image
- Dockerfile : commandes pour créer l'image Docker pour ce service uniquement
- main.py : script principal contenant la logique d'entrainement global 
- train_model.py  : script contenant la logique d'entrainement des modèles
- up.sh : script permettant de lancer un conteneur docker pour ce service uniquement. A lancer depuis la racine du projet.
- requirements.txt : contient les dépendences requis pour créer l'image 

3. Service de test

Ici, nous utilisons les données d'entraînement pour tester l'usage du modèle.
`Package` : src/models/predict
Contient les fichiers : 
- .dockerignore : fichier précisant les fichiers et répertoires à ignorer lors de la construction de l'image
- Dockerfile : commandes pour créer l'image Docker pour ce service uniquement
- predic.py : script principal contenant la logique de test du modèle en utilisant le modèle entraîné et sauvegardé sur le service d'entraînement
- up.sh : script permettant de lancer un conteneur docker pour ce service uniquement. A lancer depuis la racine du projet.
- requirements.txt : contient les dépendences requis pour créer l'image 

4. Service de serving

Ici, nous créons une application permettant d'intérroger le modèle via API en utilisant FastAPI.
`Package` : src/models/serve
Contient les fichiers : 
- .dockerignore : fichier précisant les fichiers et répertoires à ignorer lors de la construction de l'image
- Dockerfile : commandes pour créer l'image Docker pour ce service uniquement
- predic_logic.py : wrapper créé pour contenir les fonctions requis pour effectuer les prédictions (traitement de l'image, traitement des données textuelles)
- serve_model_fastapi.py : script principal contenant l'application FastAPI (les endpoints de prédiction)
- up.sh : script permettant de lancer un conteneur docker pour ce service uniquement. A lancer depuis la racine du projet.
- requirements.txt : contient les dépendences requis pour créer l'image

### API de prédiction
Le service de prédiction est composé pour le moment de deux endpoints : 
- `/status` : pour obtenir le statut de l'application. Retourne un message si l'application est up.
- `/predict` :  pour obtenir une classifiction d'un code produit en fonction des informations suivantes : </br>
`product_identifier`: str = Form(...), # Un identifiant entier pour le produit. Cet identifiant est utilisé pour associer le produit à son code de type de produit correspondant. </br>
`designation`: str = Form(...), # Le titre du produit, un court texte résumant le produit </br>
`description`: str = Form(...),  # Un texte plus détaillé décrivant le produit. Tous les commerçants n'utilisent pas ce champ, donc, afin de conserver l’originalité des données, le champ description peut contenir des valeurs NaN pour de nombreux produits. </br>
`product_id`: str = Form (...), # Un identifiant unique pour ce produit </br>
`imageid`: str = Form (...), # Un identifinat unique de l'image associé à ce produit. </br>
`image`: UploadFile = File(...)  # L'image correspondant au produit </br>

![Endpoint /predict](https://github.com/ndiguesene/Projet_Formation_MLOps_DataScientest_2025/blob/awa/restructure_folders/reports/predict_endpoint_input.png)
![Retour /predict](https://github.com/ndiguesene/Projet_Formation_MLOps_DataScientest_2025/blob/awa/restructure_folders/reports/predict_endpoint_return.png)