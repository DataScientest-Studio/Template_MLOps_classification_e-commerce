from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime
import os
from dotenv import load_dotenv
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
from airflow.utils.task_group import TaskGroup

load_dotenv(dotenv_path="/opt/airflow/mlops-project/.env")

with DAG(
    dag_id="ml_pipeline_dvc",
    start_date=datetime(2025, 6, 23),
    schedule=None,
    catchup=False
) as dag:
   
    with TaskGroup("data_stage") as data_stage:
            # -*- we are building the image for the data service to fetch data from the remote server and store it in the local filesystem -*-
        build_image_fetch_data = BashOperator(
            task_id="build_image_fetch_data",
            bash_command="""
                cd /opt/airflow/mlops-project && \
                docker build -f ./src/data/build_data/Dockerfile -t data_service .
            """
        )

    # -*- we are creating the actual container to fetch data from the remote server and store it in the local filesystem -*-. Note: The DockerOperator is used to run the container that fetches data.
        fetch_data = DockerOperator(
            task_id="fetch_data",
            image="data_service",
            container_name="data_service_container",
            api_version="auto",
            environment={
                "DATA_PATH": os.getenv("DATA_PATH"),
                "IMPORT_DATA_LOGGER_PATH": os.getenv("IMPORT_DATA_LOGGER_PATH"),
                "PYTHONUNBUFFERED": "1"
            },
            docker_url="unix://var/run/docker.sock",
            mounts=[Mount(
                source=os.getenv("PROJECT_HOST_PATH"),
                target='/app',
                type='bind'
            )
            ],
            auto_remove='success'
            )

    # -*- here we are pushing data to dvc -*-.
        push_input_data = BashOperator(
            task_id="push_input_data",
            bash_command="""
 
                cd /opt/airflow/mlops-project && \
        
                # Versionner les données d'entrée
                dvc add ./data/raw/*.csv && \
                git add ./data/raw/*.dvc && \
                dvc push
 
            """
      )
        build_image_fetch_data >> fetch_data >> push_input_data


    with TaskGroup("training_stage") as training_stage:
        # -*- we are training the model using the training service container -*-. Note: The DockerOperator is used to run the container that trains the model.
        create_model_training_image = BashOperator(
            task_id="create_model_training_image",
            bash_command="""
        
                cd /opt/airflow/mlops-project && \
                docker build -f ./src/models/train/Dockerfile -t training_service .
 
            """
        )

    # -*- we are creating the actual container to train the model -*-. Note: The DockerOperator is used to run the container that trains the model.
    # -*- under the hood, the trained model and his artifacts are pushed to MLflow in Dagshub remote storage -*-.
        train_model = DockerOperator(
            task_id="train_model",
            image="training_service",
            container_name="training_service_container",
            api_version="auto",
            environment={
                "DATA_PATH": os.getenv("DATA_PATH"),
                "TRAIN_MODEL_LOGGER_PATH": os.getenv("TRAIN_MODEL_LOGGER_PATH"),
                "TOKENIZER_CONFIG_PATH": os.getenv("TOKENIZER_CONFIG_PATH"),
                "LSTM_MODEL_PATH" : os.getenv("LSTM_MODEL_PATH"),
                "VGG16_MODEL_PATH" : os.getenv("VGG16_MODEL_PATH"),
                "BEST_WEIGHTS_PATH_PKL" : os.getenv("BEST_WEIGHTS_PATH_PKL"),
                "IMAGES_PATH":os.getenv("IMAGES_PATH"),
                "CONCATENATED_MODEL_PATH": os.getenv("CONCATENATED_MODEL_PATH"),
                "MAPPER_PATH": os.getenv("MAPPER_PATH"),
                "BEST_WEIGHTS_PATH" : os.getenv("BEST_WEIGHTS_PATH"),
                "MLFLOW_TRACKING_URI" : os.getenv("MLFLOW_TRACKING_URI"),
                "MLFLOW_EXPERIMENT_NAME": os.getenv("MLFLOW_EXPERIMENT_NAME"),
                "DAGSHUB_USERNAME":os.getenv("DAGSHUB_USERNAME"),
                "DAGSHUB_TOKEN":os.getenv("DAGSHUB_TOKEN"),
                "PYTHONUNBUFFERED": 1
            },
            mounts=[Mount(
                source=os.getenv("PROJECT_HOST_PATH"),
                target='/app',
                type='bind'
            )
            ],
            docker_url="unix://var/run/docker.sock",
            auto_remove='success'
        )

    # -*- we are pushing the model artifacts to DVC remote -*-. Note: The BashOperator is used to run the commands that push the model artifacts to DVC remote.
        push_model_artifacts = BashOperator(
            task_id="push_model_artifacts",
            bash_command="""
 
                cd /opt/airflow/mlops-project && \
        
                # Versionner les artefacts produits
                dvc add models/*.h5 && \
        
                dvc add models/*.pkl && \
                dvc add models/*.json && \
 
               # Pousser vers DVC remote
               git add models/*.dvc && \
               dvc push
 
            """
        )
        create_model_training_image >> train_model >> push_model_artifacts

    with TaskGroup("prediction_stage") as prediction_stage:
        
    # -*- we are creating the image to run a test on the model -*-. Note: The BashOperator is used to run the commands that build the image for the prediction service.
        create_prediction_image = BashOperator(
            task_id="create_model_test_image",
            bash_command="""
 
                cd /opt/airflow/mlops-project && \
                docker build -f ./src/models/predict/Dockerfile -t prediction_service .
 
            """
       )
    
    # -*- we are creating the actual container to run a test on the model -*-. Note: The DockerOperator is used to run the container that makes predictions.
        run_prediction = DockerOperator(
            task_id="run_prediction",
            image="prediction_service",
            container_name="prediction_service_container",
            api_version="auto",
            environment={
                "TOKENIZER_CONFIG_PATH": os.getenv("TOKENIZER_CONFIG_PATH"),
                "LSTM_MODEL_PATH": os.getenv("LSTM_MODEL_PATH"),
                "VGG16_MODEL_PATH": os.getenv("VGG16_MODEL_PATH"),
                "BEST_WEIGHTS_PATH": os.getenv("BEST_WEIGHTS_PATH"),
                "MAPPER_PATH": os.getenv("MAPPER_PATH"),
                "DATASET_PATH": os.getenv("DATASET_PATH"),
                "IMAGES_PATH": os.getenv("IMAGES_PATH"),
                "PREDICTIONS_PATH": os.getenv("PREDICTIONS_PATH"),
                "TEST_MODEL_LOGGER_PATH": os.getenv("TEST_MODEL_LOGGER_PATH"),
                "PYTHONUNBUFFERED": 1
            },
            mounts=[Mount(
                source=os.getenv("PROJECT_HOST_PATH"),
                target='/app',
                type='bind'
            )
            ],
            docker_url="unix://var/run/docker.sock",
            auto_remove='success'
        )
    
    # -*- we are pushing the test results to DVC remote -*-. Note: The BashOperator is used to run the commands that push the model artifacts to DVC remote.
        push_prediction_results = BashOperator(
            task_id="push_prediction_results",
            bash_command="""
 
                cd /opt/airflow/mlops-project && \
        
                # Versionner les artefacts produits
                dvc add data/predictions/predictions.json && \
 
                # Pousser vers DVC remote
                git add data/predictions/predictions.json.dvc && \
                dvc push
 
            """
        )

        create_prediction_image >> run_prediction >> push_prediction_results

    with TaskGroup("serving_stage") as serving_stage:
         # -*- we are creating the image to run an authentication image -*-. Note: The BashOperator is used to run the commands that build the image.
        create_auth_image = BashOperator(
            task_id="create_auth_image",
            bash_command="""
 
                cd /opt/airflow/mlops-project && \
                docker build -f ./src/auth_service/Dockerfile -t auth_service .
 
            """
       )

        start_auth = BashOperator(
            task_id="start_auth",
            bash_command=(
                f"docker run -d --network product_classier "
                f"--name auth_service_container "
                f"--env SECRET_KEY='{os.getenv('SECRET_KEY')}' "
                f"--env ALGORITHM='{os.getenv('ALGORITHM')}' "
                f"--env ACCESS_TOKEN_EXPIRE_MINUTES='{os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES')}' "
                f"--env AUTH_SERVICE_LOGGER_PATH='{os.getenv('AUTH_SERVICE_LOGGER_PATH')}' "
                f"--env PYTHONUNBUFFERED=1 "
                f"-v {os.getenv('PROJECT_HOST_PATH')}:/app "
                f"-p 8011:8011 "
                f"auth_service"
            )
        )
 
    # -*- we are creating the image to run an authentication container -*-. Note: The BashOperator is used to run the commands that build the image.
        create_serving_image = BashOperator(
            task_id="create_serving_image",
            bash_command="""
 
                cd /opt/airflow/mlops-project && \
                docker build -f ./src/models/serve/Dockerfile -t serving_service .
 
            """
       )

        start_serving_service = BashOperator(
            task_id="start_serving_service",
            bash_command=(
                f"docker run -d --network product_classier "
                f"--name serving_service_container "
                f"--env MAPPER_PATH='{os.getenv('MAPPER_PATH')}' "
                f"--env BEST_WEIGHTS_PATH='{os.getenv('BEST_WEIGHTS_PATH')}' "
                f"--env VGG16_MODEL_PATH='{os.getenv('VGG16_MODEL_PATH')}' "
                f"--env LSTM_MODEL_PATH='{os.getenv('LSTM_MODEL_PATH')}' "
                f"--env TOKENIZER_CONFIG_PATH='{os.getenv('TOKENIZER_CONFIG_PATH')}' "
                f"--env CONCATENATED_MODEL_PATH='{os.getenv('CONCATENATED_MODEL_PATH')}' "
                f"--env SERVING_LOGGER_PATH='{os.getenv('SERVING_LOGGER_PATH')}' "
                f"--env PYTHONUNBUFFERED=1 "
                f"-v {os.getenv('PROJECT_HOST_PATH')}:/app "
                f"-p 8000:8000 "
                f"serving_service"
            )
        )

        create_auth_image >> start_auth >> create_serving_image >> start_serving_service

    data_stage >> training_stage >> prediction_stage >> serving_stage

#build_image_fetch_data >> fetch_data >> push_input_data >> create_model_training_image >> train_model >> push_model_artifacts >> create_prediction_image >> run_prediction >> push_prediction_results >> create_auth_image >> start_auth >> create_serving_image >> start_serving_service
#>> create_model_training_image >> train_model >> push_model_artifacts >> create_prediction_image >> run_prediction >> push_prediction_results >> create_auth_image >> start_auth >> create_serving_image >> start_serving_service
