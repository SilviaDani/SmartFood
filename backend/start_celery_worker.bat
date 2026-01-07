@echo off
REM Script per avviare il worker Celery su Windows

REM Carica le variabili d'ambiente da .env (richiede pip install python-dotenv)
python -c "from dotenv import load_dotenv; load_dotenv('.env')"

REM Avvia il worker Celery
celery -A smartfood.celery_app worker ^
    --loglevel=info ^
    --concurrency=1 ^
    --max-tasks-per-child=100 ^
    --time-limit=7200 ^
    --soft-time-limit=3600

pause
