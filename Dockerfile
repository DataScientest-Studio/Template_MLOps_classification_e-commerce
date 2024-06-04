FROM python:3-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8001
ENV AWS_CONFIG_PATH=/app/.aws/.aws_config
ENV DATA_PATH=/app/data/
ENV S3_INIT_DB_PATH=rakuten_init.duckdb
ENV RAKUTEN_DB_NAME=rakuten_db.duckdb
ENV S3_BUCKET=rakutenprojectbucket

EXPOSE 8001

CMD [ "python", "./src/api/fastapi_main.py" ]