import requests
from fastapi import HTTPException, status, Depends

AUTH_SERVICE_URL = "http://auth_service_container:8011"  # URL of the authentication service

def verify_token(token: str):
    """
    Call the auth_service to verify the token.
    """
    try:
        response = requests.get(f"{AUTH_SERVICE_URL}/verify-token", params={"token": token})
        if response.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
            )
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service is unavailable",
        )

def get_token(username: str, password: str):
    """
    Call the auth_service to get a token.
    """
    try:
        response = requests.post(f"{AUTH_SERVICE_URL}/token", 
                                 data={"username": username, "password": password},
                                 headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        if response.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
            )
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service is unavailable",
        )