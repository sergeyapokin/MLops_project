FROM ghcr.io/mlflow/mlflow:latest
WORKDIR /workspace
COPY requirements.txt /workspace
RUN pip3 install --upgrade pip -r requirements.txt
COPY . /workspace
