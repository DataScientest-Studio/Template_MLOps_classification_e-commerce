from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get('/')
def get_index():
    return {'data': 'hello world'}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)