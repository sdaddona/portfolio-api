# Usa una base Python 3.11 stabile
FROM python:3.11-slim

# Evitiamo bytecode inutili e output silenziato
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Aggiorna apt e installa dipendenze di sistema minime
# - build-essential NON ci serve perché NON vogliamo compilare numpy/pandas
#   (abbiamo pin che hanno wheel manylinux ready)
# - però matplotlib ha bisogno delle lib grafiche base (freetype, libpng)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfreetype6 \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

# Crea dir app
WORKDIR /app

# Copia requirements e installa
COPY requirements.txt /app/requirements.txt

# Aggiorna pip (utile per wheel manylinux) e installa deps
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia il resto del codice applicativo
COPY server.py /app/server.py
COPY portfolio_core.py /app/portfolio_core.py
COPY portfolio_analysis_web.py /app/portfolio_analysis_web.py

# Porta esposta (Render legge $PORT comunque, ma è buona pratica)
EXPOSE 8000

# Comando di avvio:
# Render setta PORT come env var.
# Usiamo quella a runtime per il bind.
CMD exec gunicorn server:app --bind 0.0.0.0:${PORT:-8000} --workers 2 --threads 4 --timeout 120
