# Run at projet root level (important !)

# build the image
docker build -t model_builder -f src/models/train/Dockerfile .

# create the container : replace the `absolute` path to models/ and data/ directories with your own

docker run --name model_builder_container \
    --env-file .env \
    -v /Users/tiam028713/Documents/Formations/B2B_DataScientest_2025/Projet_Formation_MLOps_DataScientest_2025/models:/app/models \
    -v /Users/tiam028713/Documents/Formations/B2B_DataScientest_2025/Projet_Formation_MLOps_DataScientest_2025/data:/app/data \
    -v /Users/tiam028713/Documents/Formations/B2B_DataScientest_2025/Projet_Formation_MLOps_DataScientest_2025/logs:/app/logs \
    model_builder