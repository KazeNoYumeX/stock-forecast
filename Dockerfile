FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY config.py data.py chips.py model.py backtest.py run.py test_run.py ./
ENV DATA_DIR=/app/data PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "run.py"]
