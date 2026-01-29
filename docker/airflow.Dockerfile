FROM apache/airflow:2.7.3-python3.11

USER root

# Install system dependencies including Java for Spark
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libpq-dev \
    gcc \
    g++ \
    curl \
    openjdk-17-jre-headless \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME for PySpark
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

USER airflow

# Copy and install Python dependencies first (for better layer caching)
COPY --chown=airflow:root docker/requirements/airflow-requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Set working directory
WORKDIR /opt/airflow

# Copy application code in optimal order (least to most frequently changed)
# 1. Config files (rarely change)
COPY --chown=airflow:root config /opt/airflow/config

# 2. Common modules (models, io, data pipelines)
COPY --chown=airflow:root backend/common /opt/airflow/common

# 3. DAGs (moderate change frequency)
COPY --chown=airflow:root backend/airflow/dags /opt/airflow/dags

# Create necessary directories with proper permissions
RUN mkdir -p /opt/airflow/logs \
             /opt/airflow/plugins \
             /opt/airflow/data/raw \
             /opt/airflow/data/processed

# Set PYTHONPATH for proper module resolution
ENV PYTHONPATH=/opt/airflow:$PYTHONPATH

# Spark environment variables
ENV SPARK_HOME=/home/airflow/.local/lib/python3.11/site-packages/pyspark
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

# Enable Spark processing by default
ENV USE_SPARK=true

# Expose Airflow webserver port only (API is now separate)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl --fail http://localhost:8080/health || exit 1

# Default command will be overridden in docker-compose
