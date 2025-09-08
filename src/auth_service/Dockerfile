FROM python:3.10-slim

RUN apt-get update && apt-get install python3-pip -y python3-dev

WORKDIR /app

# Set PYTHONPATH to include /app
ENV PYTHONPATH=/app

# Create the logs directory
RUN mkdir -p /app/logs

# Create the src/data directory structure : -p flag ensures that Docker creates any missing parent directories
RUN mkdir -p src/auth_service

# Copy the Python scripts into the container
COPY src/auth_service ./src/auth_service

RUN pip install --no-cache-dir -r ./src/auth_service/requirements.txt

# Copy setup to find packages
COPY setup.py .

# Install the package inside the container
RUN pip3 install -e .

# Expose the port for FastAPI
EXPOSE 8011

CMD ["uvicorn", "src.auth_service.main:app", "--host", "0.0.0.0", "--port", "8011"]