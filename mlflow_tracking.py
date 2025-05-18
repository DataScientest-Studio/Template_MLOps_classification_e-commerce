# -------------------------------------------
# 1. Importation des bibliothèques nécessaires
# -------------------------------------------
import dagshub
import mlflow

# -------------------------------------------
# 2. Initialisation de DagsHub pour suivre les expériences via MLflow
# -------------------------------------------
dagshub.init(repo_owner='mariamanadia',
             repo_name='Projet_Formation_MLOps_DataScientest_2025',
             mlflow=True)

# -------------------------------------------
# 3. Début du suivi d'une expérience MLflow
# -------------------------------------------
mlflow.set_experiment("tracking_demo")


with mlflow.start_run():
    # Exemple de logging de paramètres
    mlflow.log_param('learning_rate', 0.01)
    mlflow.log_param('optimizer', 'adam')

    # Exemple de logging de métriques
    mlflow.log_metric('accuracy', 0.93)
    mlflow.log_metric('loss', 0.12)

    # Exemple de log de texte ou artefacts
    with open("result.txt", "w") as f:
        f.write("Résultats de l'expérience")
    mlflow.log_artifact("result.txt")

print("🎉 Tracking MLflow terminé avec succès et visible sur DagsHub !")

