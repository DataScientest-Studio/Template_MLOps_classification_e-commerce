from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import List
import duckdb
import uvicorn
import os
from aws_utils.make_db import download_db_from_s3

# Model for listing
class Listing(BaseModel):
    description: str
    designation: str
    user_prdtypecode: int
    imageid: int
    
# Define OAuth2 password bearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Password verification function
def authenticate_user(username: str, password: str):
    """
    Authenticate a user with username and password.

    Args:
        username (str): The username of the user.
        password (str): The password of the user.

    Returns:
        bool: True if the user is authenticated, False otherwise.
    """
    cursor = conn.execute(f"SELECT * FROM dim_user WHERE username = '{username}' AND hashed_password = '{password}'")
    result = cursor.fetchone()
    if not result:
        return False
    return True

# Initialize FastAPI app
app = FastAPI()

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
    
    cols = ["designation", "description", 
            "user_prdtypecode", "model_prdtypecode", 
            "waiting_datetime","validate_datetime",
            "status","user","imageid"]
    columns_str = ", ".join(cols)
    cursor = conn.execute(f"SELECT {columns_str} FROM fact_listings WHERE listing_id = {listing_id}")
    result = cursor.fetchone()
    if not result:
        raise HTTPException(status_code=404, detail="Listing not found")
    
    response = dict(zip(cols,result))
    
    return response

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
    # Logic to check if user has permission to delete listing
    cursor = conn.execute(f"SELECT user FROM fact_listings WHERE listing_id = {listing_id}")
    result = cursor.fetchone()
    if not result:
        raise HTTPException(status_code=404, detail="Listing not found")
    if result[0] != current_user["username"]:
        raise HTTPException(status_code=403, detail="This user is not the owner of this listing_id")
    # Logic to delete listing
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
    sql = f"""
            INSERT INTO fact_listings (listing_id, description, user)
            SELECT IFNULL(MAX(listing_id), 0) + 1, '{listing.description}', '{current_user["username"]}'
            FROM fact_listings;
            SELECT MAX(listing_id) FROM fact_listings;
        """ 
    
    listing_id_added = conn.execute(sql).fetchall()[0][0]

    return {"message": f"Listing {listing_id_added} added successfully"}

if __name__ == "__main__":
    
    duckdb_path = "/home/jc/Workspace/mar24cmlops_rakuten/data/rakuten_db.duckdb"
    s3_init_db_path = "/db/rakuten_init.duckdb"
    
    if not os.path.isfile(duckdb_path):
        print('No Database Found')
        # Since no database found for the API, download the initial database from S3
        download_db_from_s3(aws_config_path = '/home/jc/Workspace/mar24cmlops_rakuten/.aws/.aws_config', 
                            db_file_name = 'rakuten_init.duckdb', 
                            bucket_name = 'rakutenprojectbucket', 
                            destination_path = duckdb_path)
        
        print('Database Sucessfully Downloaded')
        
    # Load DuckDB connection   
    conn = duckdb.connect(database=duckdb_path, read_only=False)
    uvicorn.run(app, host="0.0.0.0", port=8001)