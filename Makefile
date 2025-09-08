# First launch of Airflow
init-airflow: 
	mkdir -p ./airflow/dags ./airflow/logs ./airflow/plugins ./airflow/config
	docker compose up airflow-init

start:
	docker compose up

stop:
	docker compose down

restart:
	docker compose up --build

airflow-logs:
	docker-compose logs airflow-webserver

del-containers-and-images:
	docker compose down --volumes --rmi all
	docker compose down --volumes --remove-orphans
	docker images | grep airflow | awk '{print $3}' | xargs docker rmi -f

free-space:
	df -h