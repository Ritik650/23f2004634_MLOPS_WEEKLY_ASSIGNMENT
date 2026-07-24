# 1. Official slim Python base image
FROM python:3.10-slim

# 2. Working directory inside the container
WORKDIR /app

# 3. Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the API code and the trained model
COPY iris_fastapi.py .
COPY model.joblib .

# 5. Expose the serving port
EXPOSE 8200

# 6. Run the API
CMD ["uvicorn", "iris_fastapi:app", "--host", "0.0.0.0", "--port", "8200"]
