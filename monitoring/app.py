# Dans le fichier fastApi importe et configure prometheus

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()

# Instrumentation de Prometheus pour surveiller l'API
Instrumentator().instrument(app).expose(app)

@app.get("/")
async def root():
    return {"message": "API de prédiction en cours d'exécution"}

@app.get("/predict")
async def predict():
    # Simule une prédiction (ajoute ici ton vrai code de prédiction)
    return {"prediction": "classe X"}

#Instrumentator().instrument(app).expose(app):
#-Instrumente l'API pour collecter les métriques.
#-Expose ces métriques sur l'endpoint /metrics.

#Vérifie que l’API fonctionne et expose bien les métriques : uvicorn main:app --host 0.0.0.0 --port 8000
