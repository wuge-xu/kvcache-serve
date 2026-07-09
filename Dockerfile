FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY app ./app
COPY benchmark ./benchmark
COPY README.md .

EXPOSE 18000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "18000"]
