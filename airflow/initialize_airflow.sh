# -*- ne pas utiliser sauf si on lance en mode standalone -*-

export AIRFLOW_HOME=/Users/tiam028713/Documents/Formations/Projet_2025_MLOps/Projet_Formation_MLOps_DataScientest_2025/airflow  
airflow db migrate

# Start Airflow in standalone mode (creates an admin user automatically, crdentials are saved /Users/tiam028713/airflow/simple_auth_manager_passwords.json.generated )
airflow standalone

#airflow users create \
#  --username admin \
#  --password admin \
#  --firstname Air \
#  --lastname Flow \
#  --role Admin \
#  --email awa.tiam@orange-sonatel.com
