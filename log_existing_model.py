import mlflow
import mlflow.keras
from tensorflow.keras.models import load_model
import dagshub

# Initialiser DagsHub (si nécessaire, assurez-vous que vous avez un compte et un repo DagsHub)
dagshub.init(repo_owner='mariamanadia', repo_name='Projet_Formation_MLOps_DataScientest_2025', mlflow=True)

# Définir l'URI du serveur MLflow
mlflow.set_tracking_uri("http://localhost:5000")

# Charger le modèle
model = load_model("models/best_lstm_model.h5")

# Démarrer un run MLflow
with mlflow.start_run(run_name="Log_Existing_LSTM_Model"):

    # Log des hyperparamètres
    mlflow.log_param("max_words", 10000)
    mlflow.log_param("max_sequence_length", 10)

    # Log des métriques
    mlflow.log_metric("train_loss", 0.18)
    mlflow.log_metric("train_accuracy", 0.82)

    # Log des artefacts
    mlflow.log_artifact("models/tokenizer_config.json")
    mlflow.log_artifact("models/mapper.pkl")

    # Log du modèle Keras
    mlflow.keras.log_model(model, "lstm_model_logged")

    # Log du modèle DVC (en s'assurant qu'il est bien suivi par DVC)
    # Log du modèle .h5 réel (DVC gère déjà le suivi via .dvc, pas besoin de re-loguer le fichier .dvc)
    mlflow.log_artifact("models/best_lstm_model.h5")

    print("Le modèle a été enregistré avec succès dans MLflow !")
