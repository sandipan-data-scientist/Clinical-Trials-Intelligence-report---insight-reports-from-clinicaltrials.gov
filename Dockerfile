FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY model_gb.pkl .
COPY scaler.pkl .
COPY le_intervention.pkl .
COPY le_domain.pkl .
COPY feature_cols.pkl .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]