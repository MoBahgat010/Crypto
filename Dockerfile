# Stage 1: Build the Python dependencies
FROM python:3.10-slim AS builder

# Set working directory inside the container
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc libpq-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt (or pyproject.toml/poetry.lock if using Poetry)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Run the FastAPI application
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy installed dependencies from the builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the rest of the application code
COPY . .

# Expose the port that Uvicorn will listen on
EXPOSE 8000

# Command to run the FastAPI app with Uvicorn
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]