# Run at projet root level (important !)

# build the image
docker build -t importer_image -f src/data/build_data/Dockerfile .

# create the container : replace the `absolute` path to logs/ and data/ directories with your own

docker run --name data_importer \
    --env-file .env \
    -v /Users/tiam028713/Documents/Formations/B2B_DataScientest_2025/Projet_Formation_MLOps_DataScientest_2025/data:/app/data \
    -v /Users/tiam028713/Documents/Formations/B2B_DataScientest_2025/Projet_Formation_MLOps_DataScientest_2025/logs:/app/logs \
    importer_image