import mlflow
import mlflow.keras
from tensorflow.keras.models import load_model
import mlflow.pyfunc

def log_model_to_mlflow(model_path, run_name="LSTM_Model_Run"):
    """
    Fonction pour charger un modèle LSTM et l'enregistrer dans MLflow.
    
    Args:
    - model_path: Chemin vers le modèle Keras à charger
    - run_name: Nom de l'exécution MLflow (facultatif)
    """
    
    # Charger le modèle
    model = load_model(model_path)

    # Démarrer un suivi avec MLflow
    with mlflow.start_run(run_name=run_name):
        # Enregistrer le modèle LSTM dans MLflow
        mlflow.keras.log_model(model, "lstm_model")

        # Log des hyperparamètres
        mlflow.log_param("max_words", 10000)  # Exemple d'hyperparamètre à ajuster
        mlflow.log_param("max_sequence_length", 10)  # Exemple d'hyperparamètre à ajuster
        
        # Log des métriques
        mlflow.log_metric("train_loss", 0.18)  # Exemple, à remplacer par des métriques réelles
        mlflow.log_metric("train_accuracy", 0.82)  # Exemple, à remplacer par des métriques réelles

        # Log des artefacts (fichiers supplémentaires)
        mlflow.log_artifact("models/tokenizer_config.json", "tokenizer_config.json")
        mlflow.log_artifact("models/mapper.pkl", "mapper.pkl")

        print("Le modèle a été enregistré avec succès dans MLflow !")

# Appel de la fonction pour enregistrer le modèle
if __name__ == "__main__":
    model_path = "models/best_lstm_model.h5"
    log_model_to_mlflow(model_path)

