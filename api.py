from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def status():
    return {"message": "Bienvenue"}

@app.get("/status")
def status():
    return {"status": "L'API fonctionne correctement"}

@app.post("/predict")
def predict(data: dict):
    # Prédiction avec ton modèle ici
    sample_text = {"predicted": "1", "label": "label_1"}
    return sample_text
