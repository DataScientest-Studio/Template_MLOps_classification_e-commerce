from fastapi import FastAPI, File, UploadFile, Form, Request, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import List
import json
from tensorflow import keras
import pandas as pd
import os
from dotenv import load_dotenv
from uuid import uuid4
import time
import logging
from logging.handlers import RotatingFileHandler
from logging import Formatter
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse
from slowapi.middleware import SlowAPIMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, generate_latest
from starlette.responses import Response
from src.models.serve.predict_logic import make_prediction, preprocess_image
from src.models.serve.auth_utils import verify_token, get_token


load_dotenv()

app = FastAPI()

logger = None  # global logger variable

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

class SubPrediction(BaseModel):
    label: str
    confidence: float
    prediction: int


class PredictionResponse(BaseModel):
    predictions: List[SubPrediction]
    message: str

   # Logger setup
logger = logging.getLogger(__name__)
log_file_path = os.path.join("logs", "api.log")

os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
fileHandler = RotatingFileHandler(log_file_path, maxBytes=5 * 1024 * 1024, backupCount=3)
formatter = Formatter("%(asctime)s [%(levelname)s] %(message)s")
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)
logger.setLevel(logging.INFO) 

@app.on_event("startup")
async def load_models():
    global tokenizer, lstm, vgg16, best_weights, mapper

    #logger = setup_logger()
    tokenizer_config_path = os.getenv("TOKENIZER_CONFIG_PATH", "models/tokenizer_config.json")
    lstm_model_path = os.getenv("LSTM_MODEL_PATH", "models/best_lstm_model.h5")
    vgg16_model_path = os.getenv("VGG16_MODEL_PATH", "models/best_vgg16_model.h5")
    best_weights_path = os.getenv("BEST_WEIGHTS_PATH", "models/best_weights.json")
    mapper_path = os.getenv("MAPPER_PATH", "models/mapper.json")

    try:
        # Vérifications fichiers existants
        if not os.path.isfile(tokenizer_config_path):
            raise FileNotFoundError(f"Tokenizer config file not found: {tokenizer_config_path}")
        if not os.path.isfile(lstm_model_path):
            raise FileNotFoundError(f"LSTM model file not found: {lstm_model_path}")
        if not os.path.isfile(vgg16_model_path):
            raise FileNotFoundError(f"VGG16 model file not found: {vgg16_model_path}")
        if not os.path.isfile(best_weights_path):
            raise FileNotFoundError(f"Best weights file not found: {best_weights_path}")
        if not os.path.isfile(mapper_path):
            raise FileNotFoundError(f"Mapper file not found: {mapper_path}")

        # Charger tokenizer (lire le JSON sous forme de chaîne)
        with open(tokenizer_config_path, "r", encoding="utf-8") as f:
            tokenizer_json_str = f.read()
        tokenizer = keras.preprocessing.text.tokenizer_from_json(tokenizer_json_str)

        # Charger les modèles Keras
        lstm = keras.models.load_model(lstm_model_path)
        vgg16 = keras.models.load_model(vgg16_model_path)

        # Charger les poids (json)
        with open(best_weights_path, "r", encoding="utf-8") as f:
            best_weights = json.load(f)

        # Charger le mapper (json)
        with open(mapper_path, "r", encoding="utf-8") as f:
            mapper = json.load(f)

        logger.info("✅ Models, tokenizer and configs loaded successfully.")
    except Exception as e:
        logger.exception(f"❌ Error during model loading: {e}")
        raise RuntimeError("Startup failed: model loading error.")

@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"message": "Rate limit exceeded. Please try again later."},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "An internal server error occurred."},
    )


@app.get("/status")
@limiter.limit("5/minute")
def get_status(request: Request):
    """
    Endpoint to check the status of the service.
    """
    return {"message": "Model Serving via FastAPI is running !"}


@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    token = get_token(form_data.username, form_data.password)
    return token


@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("10/minute")
async def predict_code(
    request: Request,
    product_identifier: str = Form(...),
    designation: str = Form(...),
    description: str = Form(...),
    product_id: str = Form(...),
    imageid: str = Form(...),
    image: UploadFile = File(...),
    token: dict = Depends(verify_token),
):
    current_user = token
    logger.info(f"Prediction requested by user: {current_user.get('username', 'unknown')}")

    image_data = await image.read()
    processed_image = preprocess_image(image_data)

    df = pd.DataFrame({
        "product_identifier": [product_identifier],
        "description": [description],
        "designation": [designation],
        "product_id": [product_id],
        "imageid": [imageid]
    })

    try:
        predictions = make_prediction(df, processed_image, tokenizer, lstm, vgg16, best_weights, mapper)
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during prediction.")

    sub_predictions = [SubPrediction(**prediction) for prediction in predictions]

    return {
        "predictions": sub_predictions,
        "message": "Prediction successful"
    }


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid4())
    logger.info(f"Request ID={request_id} - Start request: {request.method} {request.url}")
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    logger.info(
        f"Request ID={request_id} - Completed request: {request.method} {request.url} "
        f"Status code={response.status_code} - Process time={process_time:.4f}s"
    )
    return response


instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)
