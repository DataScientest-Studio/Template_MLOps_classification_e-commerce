from fastapi import FastAPI, Depends, HTTPException, status, Form
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta
from src.auth_service.security_logic import authenticate_user, create_access_token, get_current_user
from pydantic import BaseModel

app = FastAPI()

class TokenRequest(BaseModel):
    username: str
    password: str

# Securing API 3 : Token endpoint
@app.post("/token")
async def login_for_access_token(
    username: str = Form(...),
    password: str = Form(...)
    ):
    """
    Endpoint to authenticate a user and return a JWT token.
    """
    user = authenticate_user(username, password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/verify-token")
def verify_token(token: str):
    """
    Endpoint to verify the validity of a JWT token.
    """
    user = get_current_user(token)
    return {"username": user["username"], "full_name": user["full_name"]}