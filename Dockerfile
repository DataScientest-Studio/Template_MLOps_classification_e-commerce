FROM python:3-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8001

ENV ECS_SERVICE: rakutenapi
ENV ECS_CLUSTER: rakutenproject
ENV ECS_TASK_DEFINITION: ./aws/task-definition.json
ENV CONTAINER_NAME: "app"      

EXPOSE 8001
CMD [ "python", "./src/api/fastapi_main.py" ]