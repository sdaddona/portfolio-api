FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libfreetype6 \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY server.py /app/server.py
COPY portfolio_core.py /app/portfolio_core.py
COPY portfolio_analysis_web.py /app/portfolio_analysis_web.py

EXPOSE 8000

CMD exec gunicorn server:app --bind 0.0.0.0:${PORT:-8000} --workers 2 --threads 4 --timeout 120
