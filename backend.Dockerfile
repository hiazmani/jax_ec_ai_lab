FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project
COPY . .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install -e .

EXPOSE 5001

# Set PYTHONPATH to include the server directory for config_preconfigured
ENV PYTHONPATH="/app/server:${PYTHONPATH}"

CMD ["python", "server/main.py"]
