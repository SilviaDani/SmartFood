#!/bin/bash
# Script per avviare il worker Celery

# Assicurati di essere nella directory del backend
cd "$(dirname "$0")"

# Carica le variabili d'ambiente
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

# Avvia il worker Celery
# -A: app
# -l: log level
# -c: concurrency (numero di worker, 1 per GPU)
# --loglevel: log level
# -P: pool type (solo se Flask-Celery disponibile)

celery -A smartfood.celery_app worker \
    --loglevel=info \
    --concurrency=1 \
    --max-tasks-per-child=100 \
    --time-limit=7200 \
    --soft-time-limit=3600
