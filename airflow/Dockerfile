FROM apache/airflow:3.0.2
USER root
RUN apt-get update ; apt-get update && apt-get install -y gosu
USER airflow
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt