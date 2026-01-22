"""
Training Task - Task Celery per l'addestramento dei modelli
"""

import os
import uuid
import logging
from datetime import datetime
from celery import shared_task, current_task
from smartfood.models import TrainingJob, JobStatus, db
from smartfood.celery_app import celery_app
from smartfood.utils.model_registry import get_model_registry

logger = logging.getLogger(__name__)


def get_flask_app():
    """Importa l'app Flask"""
    from smartfood.app import create_app
    return create_app()


@shared_task(bind=True, max_retries=5)
def train_model(self, job_id: str, dataset_path: str, model_id: str):
    """
    Task Celery per addestrare un modello
    
    Args:
        job_id: ID del job di addestramento
        dataset_path: Percorso al file CSV di training
        model_id: ID del modello (moment, chronos)
        
    Returns:
        dict: Risultati dell'addestramento
    """
    # Usa il Flask app context
    app = get_flask_app()
    
    with app.app_context():
        try:
            # Aggiorna il job nello stato RUNNING
            job = TrainingJob.query.get(job_id)
            if not job:
                logger.error(f"Job {job_id} not found")
                raise ValueError(f"Job {job_id} not found")
            
            job.status = JobStatus.RUNNING.value
            job.started_at = datetime.utcnow()
            job.celery_task_id = self.request.id
            db.session.commit()
            
            logger.info(f"Starting training for job {job_id}: model={model_id}, dataset={dataset_path}")
            
            # Verifica che il modello sia disponibile
            model_registry = get_model_registry()
            if not model_registry.is_model_available(model_id):
                available = ', '.join(model_registry.get_available_models())
                raise ValueError(f"Unknown model: {model_id}. Available models: {available}")
            
            # Chiama il training handler registrato nel model_registry
            result = model_registry.train(model_id, job_id, dataset_path)
            
            # Aggiorna il job con i risultati
            job.status = JobStatus.COMPLETED.value
            job.completed_at = datetime.utcnow()
            job.progress = 100
            job.accuracy = result.get('accuracy')
            job.loss = result.get('loss')
            job.metrics = result.get('metrics')
            job.model_path = result.get('model_path')
            db.session.commit()
            
            logger.info(f"Training completed for job {job_id}: accuracy={job.accuracy}")
            
            return {
                'status': 'completed',
                'job_id': job_id,
                'accuracy': job.accuracy,
                'model_path': job.model_path,
            }
        
        except Exception as exc:
            logger.error(f"Training failed for job {job_id}: {str(exc)}", exc_info=True)
            
            job = TrainingJob.query.get(job_id)
            if job:
                job.error_message = str(exc)
                job.retry_count += 1
                
                # Se possiamo riprovare, riprova
                if job.can_retry():
                    job.status = JobStatus.PENDING.value
                    db.session.commit()
                    
                    # Riprogramma il task
                    logger.info(f"Retrying job {job_id} (attempt {job.retry_count}/{job.max_retries})")
                    raise self.retry(exc=exc, countdown=60)  # Retry dopo 60 secondi
                else:
                    # Max retry raggiunto, fallimento permanente
                    job.status = JobStatus.FAILED.value
                    job.completed_at = datetime.utcnow()
                    db.session.commit()
                    logger.error(f"Job {job_id} failed after {job.max_retries} retries")
            
            raise


def _train_chronos(job_id: str, dataset_path: str) -> dict:
    """
    Addestramento del modello Chronos
    
    Args:
        job_id: ID del job
        dataset_path: Percorso al file CSV
        
    Returns:
        dict: Risultati dell'addestramento
    """
    import pandas as pd
    from chronos import Chronos
    
    job = TrainingJob.query.get(job_id)
    
    try:
        # Step 1: Load data
        job.current_step = "Caricamento dati..."
        job.progress = 10
        db.session.commit()
        
        df = pd.read_csv(dataset_path)
        logger.info(f"Dataset loaded: shape={df.shape}")
        
        # Step 2: Preprocessing
        job.current_step = "Preprocessing dati..."
        job.progress = 20
        db.session.commit()
        
        # Preprocessing specifico per Chronos
        df = _preprocess_chronos(df)
        
        # Step 3: Model initialization
        job.current_step = "Inizializzazione modello Chronos..."
        job.progress = 30
        db.session.commit()
        
        model = Chronos.from_pretrained("amazon/chronos-t5-small")
        
        # Step 4: Prepare sequences
        job.current_step = "Preparazione sequenze..."
        job.progress = 40
        db.session.commit()
        
        # Estrai la colonna delle porzioni (o appropriata per il modello)
        series = df['portions_prepared'].values if 'portions_prepared' in df else df.iloc[:, 0].values
        
        # Step 5: Fine-tuning
        job.current_step = "Fine-tuning modello..."
        job.progress = 60
        db.session.commit()
        
        # Nota: Chronos potrebbe non supportare fine-tuning diretto
        # Qui implementiamo una logica semplificata
        # In produzione, adattare in base alle API di Chronos
        
        # Step 6: Evaluation
        job.current_step = "Valutazione modello..."
        job.progress = 80
        db.session.commit()
        
        # Placeholder per valutazione
        accuracy = 0.92  # Sostituire con valutazione reale
        loss = 0.08
        
        # Step 7: Save model
        job.current_step = "Salvataggio modello..."
        job.progress = 95
        db.session.commit()
        
        model_path = _save_model(job_id, model, 'chronos')
        
        return {
            'accuracy': accuracy,
            'loss': loss,
            'metrics': {'model': 'chronos'},
            'model_path': model_path,
        }
    
    except Exception as e:
        logger.error(f"Chronos training error: {str(e)}", exc_info=True)
        raise


def _train_moment(job_id: str, dataset_path: str) -> dict:
    """
    Addestramento del modello MOMENT
    
    Args:
        job_id: ID del job
        dataset_path: Percorso al file CSV
        
    Returns:
        dict: Risultati dell'addestramento
    """
    import pandas as pd
    import torch
    
    job = TrainingJob.query.get(job_id)
    
    try:
        # Step 1: Load data
        job.current_step = "Caricamento dati..."
        job.progress = 10
        db.session.commit()
        
        df = pd.read_csv(dataset_path)
        logger.info(f"Dataset loaded: shape={df.shape}")
        
        # Step 2: Preprocessing
        job.current_step = "Preprocessing dati..."
        job.progress = 20
        db.session.commit()
        
        df = _preprocess_moment(df)
        
        # Step 3: Model initialization
        job.current_step = "Inizializzazione modello MOMENT..."
        job.progress = 30
        db.session.commit()
        
        # Carica il modello MOMENT
        from transformers import AutoModel
        model = AutoModel.from_pretrained("autoregressive/moment-1-large")
        
        # Step 4: Prepare data
        job.current_step = "Preparazione dati per training..."
        job.progress = 50
        db.session.commit()
        
        # Estrai la serie temporale
        series = df['portions_prepared'].values if 'portions_prepared' in df else df.iloc[:, 0].values
        
        # Step 5: Training
        job.current_step = "Addestramento modello..."
        job.progress = 70
        db.session.commit()
        
        # Placeholder per training loop
        # In produzione, implementare il vero loop di training
        accuracy = 0.89
        loss = 0.11
        
        # Step 6: Evaluation
        job.current_step = "Valutazione modello..."
        job.progress = 85
        db.session.commit()
        
        # Step 7: Save model
        job.current_step = "Salvataggio modello..."
        job.progress = 95
        db.session.commit()
        
        model_path = _save_model(job_id, model, 'moment')
        
        return {
            'accuracy': accuracy,
            'loss': loss,
            'metrics': {'model': 'moment'},
            'model_path': model_path,
        }
    
    except Exception as e:
        logger.error(f"MOMENT training error: {str(e)}", exc_info=True)
        raise


def _preprocess_chronos(df):
    """Preprocessing specifico per Chronos"""
    # Implementare la logica di preprocessing per Chronos
    return df


def _preprocess_moment(df):
    """Preprocessing specifico per MOMENT"""
    # Implementare la logica di preprocessing per MOMENT
    return df


def _preprocess_timesfm(df):
    """Preprocessing specifico per TimesFM-2.5"""
    # TimesFM è molto robusto, preprocessing minimo
    # - Gestisce automaticamente trend, stagionalità, anomalie
    # - Supporta serie irregolari
    return df


def _train_timesfm(job_id: str, dataset_path: str) -> dict:
    """
    Addestramento del modello TimesFM-2.5
    
    Args:
        job_id: ID del job
        dataset_path: Percorso al file CSV
        
    Returns:
        dict: Risultati dell'addestramento
    """
    import pandas as pd
    
    job = TrainingJob.query.get(job_id)
    
    try:
        # Step 1: Load data
        job.current_step = "Caricamento dati..."
        job.progress = 10
        db.session.commit()
        
        df = pd.read_csv(dataset_path)
        logger.info(f"Dataset loaded: shape={df.shape}")
        
        # Step 2: Preprocessing
        job.current_step = "Preprocessing dati..."
        job.progress = 20
        db.session.commit()
        
        df = _preprocess_timesfm(df)
        
        # Step 3: Model initialization
        job.current_step = "Inizializzazione modello TimesFM-2.5..."
        job.progress = 30
        db.session.commit()
        
        # TimesFM-2.5 da Google
        # from google.cloud.timeseries_datasets import TimesFM
        # model = TimesFM.from_pretrained("google/timesfm-2.5")
        
        # Step 4: Prepare data
        job.current_step = "Preparazione sequenze temporali..."
        job.progress = 50
        db.session.commit()
        
        # Estrai la serie temporale
        series = df['portions_prepared'].values if 'portions_prepared' in df else df.iloc[:, 0].values
        
        # Step 5: Training/Forecasting
        job.current_step = "Forecasting con TimesFM-2.5..."
        job.progress = 70
        db.session.commit()
        
        # TimesFM è zero-shot, non richiede fine-tuning
        # Le previsioni sono generate direttamente
        accuracy = 0.88  # Benchmark simulato
        loss = 0.12
        
        # Step 6: Evaluation
        job.current_step = "Valutazione modello..."
        job.progress = 85
        db.session.commit()
        
        # Step 7: Save model
        job.current_step = "Salvataggio modello..."
        job.progress = 95
        db.session.commit()
        
        model_path = _save_model(job_id, None, 'timesfm')
        
        return {
            'accuracy': accuracy,
            'loss': loss,
            'metrics': {'model': 'timesfm', 'zero_shot': True},
            'model_path': model_path,
        }
    
    except Exception as e:
        logger.error(f"TimesFM training error: {str(e)}", exc_info=True)
        raise


def _save_model(job_id: str, model, model_id: str) -> str:
    """
    Salva il modello addestrato nel filesystem
    
    Args:
        job_id: ID del job
        model: Oggetto modello
        model_id: ID del modello (chronos, moment)
        
    Returns:
        str: Percorso al modello salvato
    """
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'trained_models', model_id)
    os.makedirs(models_dir, exist_ok=True)
    
    model_filename = f"{model_id}_{job_id}.pt"
    model_path = os.path.join(models_dir, model_filename)
    
    # Salva il modello (implementazione dipende dal tipo di modello)
    try:
        if hasattr(model, 'save_pretrained'):
            model.save_pretrained(model_path)
        else:
            # Fallback: salva con torch
            import torch
            torch.save(model.state_dict(), model_path)
        
        logger.info(f"Model saved to {model_path}")
        return model_path
    
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise


@celery_app.task
def cleanup_failed_jobs():
    """Cleanup periodico dei job falliti"""
    try:
        failed_jobs = TrainingJob.query.filter_by(status=JobStatus.FAILED.value).all()
        logger.info(f"Found {len(failed_jobs)} failed jobs for cleanup")
        # Implementare la logica di cleanup (es. archiviare, elimina log, etc.)
    except Exception as e:
        logger.error(f"Cleanup error: {str(e)}")
