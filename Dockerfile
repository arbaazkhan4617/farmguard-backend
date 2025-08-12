# Dockerfile
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY . /app

ENV MODEL_PATH=/app/model/saved_model
ENV UPLOAD_DIR=/app/uploads
RUN mkdir -p /app/uploads

# Uvicorn will be served by the image default

