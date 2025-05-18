import dagshub 
import mlflow
import mlflow.keras
from tensorflow.keras.models import load_model

# -------------------------------------------
# 1. Initialisation de DagsHub avec MLflow
# -------------------------------------------
dagshub.init(repo_owner='mariamanadia',
             repo_name='Projet_Formation_MLOps_DataScientest_2025',
             mlflow=True)

# -------------------------------------------
# 2. Définir l'URI du serveur MLflow (si nécessaire)
# -------------------------------------------
# Si tu utilises un serveur MLflow externe, décommente la ligne suivante et définis l'URI
# mlflow.set_tracking_uri("https://dagshub.com/mariamanadia/Projet_Formation_MLOps_DataScientest_2025/experiments")

# -------------------------------------------
# 3. Définir ou créer l'expérience MLflow
# -------------------------------------------
mlflow.set_experiment("tracking_demo")

# -------------------------------------------
# 4. Charger le modèle Keras existant
# -------------------------------------------
model = load_model("models/best_lstm_model.h5")

# -------------------------------------------
# 5. Démarrer un run MLflow pour enregistrer les hyperparamètres, métriques et modèle
# -------------------------------------------
with mlflow.start_run(run_name="Log_Existing_LSTM_Model"):

    # Log des hyperparamètres du modèle
    mlflow.log_param("max_words", 10000)
    mlflow.log_param("max_sequence_length", 10)

    # Log des métriques d'entraînement
    mlflow.log_metric("train_loss", 0.18)  # Remplace par la vraie perte de ton modèle
    mlflow.log_metric("train_accuracy", 0.82)  # Remplace par la vraie précision de ton modèle

    # Log des artefacts : fichiers associés au modèle
    mlflow.log_artifact("models/tokenizer_config.json")
    mlflow.log_artifact("models/mapper.pkl")

    # Log du modèle Keras
    mlflow.keras.log_model(model, "lstm_model_logged")

    # Log du fichier DVC (si tu utilises DVC pour le versioning des modèles)
    mlflow.log_artifact("models/best_lstm_model.h5.dvc")

    print("Le modèle a été enregistré avec succès dans MLflow et DagsHub !")
