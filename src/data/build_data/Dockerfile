FROM python:3.10-slim
  
RUN apt-get update && apt-get install python3-pip -y python3-dev && rm -rf /var/lib/apt/lists/* 
# Clean up apt cache to reduce image size

# Set the working directory
WORKDIR /app

# Create the logs directory
RUN mkdir -p /app/logs

# Create the src/data directory structure : -p flag ensures that Docker creates any missing parent directories
RUN mkdir -p src/data/build_data

# Copy the Python scripts into the container
COPY src/data/build_data/ ./src/data/build_data

# --no-cache-dir ensures that pip doesn't cache the downloaded package files, which will keep your Docker image smaller
RUN pip install --no-cache-dir -r ./src/data/build_data/requirements.txt

# Copy setup to find packages
COPY setup.py .

# Install the package inside the container
RUN pip3 install -e .


CMD ["python3", "./src/data/build_data/import_raw_data.py"]