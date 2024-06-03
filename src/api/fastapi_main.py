from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import List
import duckdb
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Define OAuth2 password bearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Load DuckDB connection
conn = duckdb.connect(database="your_duckdb_file.duckdb", read_only=False)

# Model for listing
class Listing(BaseModel):
    description: str
    user_id: str

# OAuth2 password verification function
def authenticate_user(username: str, password: str):
    """
    Authenticate a user with username and password.

    Args:
        username (str): The username of the user.
        password (str): The password of the user.

    Returns:
        bool: True if the user is authenticated, False otherwise.
    """
    cursor = conn.execute(f"SELECT * FROM dim_users WHERE username = '{username}' AND password = '{password}'")
    result = cursor.fetchone()
    if not result:
        return False
    return True

@app.get('/')
def get_index():
    return {'data': 'hello world'}

# Endpoint to authenticate and get token
@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Endpoint to authenticate and retrieve access token.

    Args:
        form_data (OAuth2PasswordRequestForm): Form data containing username and password.

    Returns:
        dict: Dictionary containing access token.
    """
    username = form_data.username
    password = form_data.password
    if not authenticate_user(username, password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    return {"access_token": username, "token_type": "bearer"}

# Dependency function to get current user from token
async def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Dependency function to get current user from OAuth2 token.

    Args:
        token (str): OAuth2 token.

    Returns:
        dict: Dictionary containing user information.
    """
    return {"username": token}

# Read listing endpoint
@app.get("/read_listing/{listing_id}")
async def read_listing(listing_id: int, current_user: dict = Depends(get_current_user)):
    """
    Endpoint to read listing description.

    Args:
        listing_id (int): The ID of the listing.
        current_user (dict): Dictionary containing current user information.

    Returns:
        dict: Dictionary containing listing description.
    """
    # Dummy logic to retrieve listing description based on listing_id
    cursor = conn.execute(f"SELECT description FROM fact_listings WHERE listing_id = {listing_id}")
    result = cursor.fetchone()
    if not result:
        raise HTTPException(status_code=404, detail="Listing not found")
    return {"listing_description": result[0]}

# Delete listing endpoint
@app.delete("/delete_listing/{listing_id}")
async def delete_listing(listing_id: int, current_user: dict = Depends(get_current_user)):
    """
    Endpoint to delete a listing.

    Args:
        listing_id (int): The ID of the listing to be deleted.
        current_user (dict): Dictionary containing current user information.

    Returns:
        dict: Message indicating success or failure.
    """
    # Dummy logic to check if user has permission to delete listing
    cursor = conn.execute(f"SELECT user_id FROM fact_listings WHERE listing_id = {listing_id}")
    result = cursor.fetchone()
    if not result:
        raise HTTPException(status_code=404, detail="Listing not found")
    if result[0] != current_user["username"]:
        raise HTTPException(status_code=403, detail="User is not authorized to delete this listing")
    # Dummy logic to delete listing
    conn.execute(f"DELETE FROM fact_listings WHERE listing_id = {listing_id}")
    return {"message": "Listing deleted successfully"}

# Add listing endpoint
@app.post("/add_listing")
async def add_listing(listing: Listing, current_user: dict = Depends(get_current_user)):
    """
    Endpoint to add a new listing.

    Args:
        listing (Listing): Listing information to be added.
        current_user (dict): Dictionary containing current user information.

    Returns:
        dict: Message indicating success or failure.
    """
    # Dummy logic to generate listing_id and insert new listing into database
    listing_id = 1  # Dummy logic to generate listing_id
    conn.execute(f"INSERT INTO fact_listings VALUES ({listing_id}, '{listing.description}', '{listing.user_id}')")
    return {"message": "Listing added successfully"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)