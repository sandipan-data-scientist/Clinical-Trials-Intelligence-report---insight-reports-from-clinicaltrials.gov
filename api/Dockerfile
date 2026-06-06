FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir fastapi uvicorn joblib scikit-learn numpy

COPY artifacts/ artifacts/
COPY api/app.py api/app.py
COPY src/ src/

EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]