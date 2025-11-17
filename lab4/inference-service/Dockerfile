FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl dnsutils && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8000

CMD ["sh", "-c", "echo '192.168.56.20 mlflow.labs.itmo.loc' >> /etc/hosts && uvicorn app:app --host 0.0.0.0 --port 8000"]
