"""
Celery Configuration - Setup per la coda di addestramento
"""

import os
from celery import Celery
from celery.schedules import crontab

# Crea l'app Celery
celery_app = Celery(__name__)

# Configurazione
celery_app.conf.update(
    # Message broker (Redis)
    broker_url=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    
    # Result backend (Redis)
    result_backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0'),
    
    # Task settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Worker settings
    worker_prefetch_multiplier=1,  # Un task alla volta per worker
    worker_max_tasks_per_child=1000,  # Ricicla il worker ogni 1000 task
    
    # Task timing
    task_soft_time_limit=3600,  # 1 ora soft timeout
    task_time_limit=7200,  # 2 ore hard timeout
    
    # Retry policy
    task_acks_late=True,  # ACK solo dopo completamento
    task_reject_on_worker_lost=True,  # Requeue se worker muore
    
    # Beat schedule (per scheduled tasks in futuro)
    beat_schedule={
        # Esempio: cleanup dei job falliti ogni ora
        # 'cleanup-failed-jobs': {
        #     'task': 'smartfood.tasks.training_task.cleanup_failed_jobs',
        #     'schedule': crontab(minute=0),  # Ogni ora
        # },
    },
)

# Import automatico dei task quando l'app Celery si inizializza
celery_app.autodiscover_tasks(['smartfood.tasks'])


@celery_app.task(bind=True)
def debug_task(self):
    """Debug task per testare Celery"""
    print(f'Request: {self.request!r}')
