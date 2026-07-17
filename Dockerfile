FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1     PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip     python -m pip install --upgrade pip &&     grep -vE '^(torch|torchvision|torchaudio)([<>=!~].*)?$' requirements.txt         > /tmp/requirements-no-torch.txt &&     python -m pip install         --index-url https://download.pytorch.org/whl/cpu         torch &&     python -m pip install         -r /tmp/requirements-no-torch.txt

COPY app ./app
COPY benchmark ./benchmark
COPY README.md .

EXPOSE 18000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "18000"]
