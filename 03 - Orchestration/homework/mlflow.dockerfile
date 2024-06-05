# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install mlflow
RUN pip install mlflow==2.12.1

# Expose the port mlflow will run on
EXPOSE 5000

# Command to run mlflow server
CMD [ \
    "mlflow", "server", \
    "--backend-store-uri", "sqlite:///home/mlflow/mlflow.db", \
    "--host", "0.0.0.0", \
    "--port", "5001" \
]