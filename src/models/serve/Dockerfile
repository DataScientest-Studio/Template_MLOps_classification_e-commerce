FROM python:3.10-slim
  
RUN apt-get update && apt-get install python3-pip -y python3-dev

# Set the working directory
WORKDIR /app

# Set PYTHONPATH to include /app
ENV PYTHONPATH=/app

# Create the logs directory
RUN mkdir -p /app/logs

# Create the src/data directory structure : -p flag ensures that Docker creates any missing parent directories
RUN mkdir -p src/models/serve

# Copy the Python scripts into the container
COPY src/models/serve ./src/models/serve

# --no-cache-dir ensures that pip doesn't cache the downloaded package files, which will keep your Docker image smaller
RUN pip3 install --no-cache-dir -r ./src/models/serve/requirements.txt

# Download NLTK resources
RUN python3 -m nltk.downloader punkt
RUN python3 -m nltk.downloader punkt_tab

# Copy setup to find packages
COPY setup.py .

# Install the package inside the container
RUN pip3 install -e .
 
# Expose the port for FastAPI
EXPOSE 8000

CMD ["uvicorn", "src.models.serve.serve_model_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]