"""
Training Job Model - Rappresenta un job di addestramento nel database
"""

from datetime import datetime
from enum import Enum
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class JobStatus(Enum):
    """Stati possibili di un job di addestramento"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingJob(db.Model):
    """
    Modello per tracciare i job di addestramento
    
    Attributi:
        id: UUID univoco del job
        model_id: ID del modello (moment, chronos)
        dataset_id: Nome del file CSV utilizzato
        status: Stato del job (pending, running, completed, failed, cancelled)
        progress: Percentuale di completamento (0-100)
        current_step: Descrizione dello step attuale
        accuracy: Accuratezza finale del modello
        error_message: Messaggio di errore se fallisce
        retry_count: Numero di retry effettuati
        max_retries: Numero massimo di retry consentiti
        scheduled_at: Data/ora per scheduling futuro (nullable)
        started_at: Data/ora di inizio addestramento
        completed_at: Data/ora di completamento
        created_at: Data/ora di creazione del job
        priority: Priorità del job (0=normale, 1=alta)
        model_path: Percorso al modello salvato
        duration: Durata dell'addestramento in secondi
    """
    
    __tablename__ = 'training_job'
    
    # Colonne primarie
    id = db.Column(db.String(36), primary_key=True)
    model_id = db.Column(db.String(50), nullable=False, index=True)
    dataset_id = db.Column(db.String(255), nullable=False)
    
    # Stato e progresso
    status = db.Column(
        db.String(20),
        default=JobStatus.PENDING.value,
        nullable=False,
        index=True
    )
    progress = db.Column(db.Integer, default=0)
    current_step = db.Column(db.String(255), nullable=True)
    
    # Risultati
    accuracy = db.Column(db.Float, nullable=True)
    loss = db.Column(db.Float, nullable=True)
    metrics = db.Column(db.JSON, nullable=True)  # Metriche aggiuntive in JSON
    
    # Errori
    error_message = db.Column(db.Text, nullable=True)
    
    # Retry logic
    retry_count = db.Column(db.Integer, default=0)
    max_retries = db.Column(db.Integer, default=5)
    
    # Timing
    scheduled_at = db.Column(db.DateTime, nullable=True)
    started_at = db.Column(db.DateTime, nullable=True)
    completed_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    # Priorità e configurazione
    priority = db.Column(db.Integer, default=0)
    
    # Output
    model_path = db.Column(db.String(255), nullable=True)
    
    # Celery
    celery_task_id = db.Column(db.String(155), nullable=True, unique=True)
    
    def __repr__(self):
        return f"<TrainingJob {self.id} - {self.model_id} - {self.status}>"
    
    def to_dict(self):
        """Converte il job a dizionario"""
        duration = None
        if self.started_at and self.completed_at:
            duration = (self.completed_at - self.started_at).total_seconds()
        elif self.started_at:
            duration = (datetime.utcnow() - self.started_at).total_seconds()
        
        return {
            'id': self.id,
            'model_id': self.model_id,
            'dataset_id': self.dataset_id,
            'status': self.status,
            'progress': self.progress,
            'current_step': self.current_step,
            'accuracy': self.accuracy,
            'loss': self.loss,
            'metrics': self.metrics,
            'error_message': self.error_message,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'scheduled_at': self.scheduled_at.isoformat() if self.scheduled_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'created_at': self.created_at.isoformat(),
            'priority': self.priority,
            'model_path': self.model_path,
            'duration': duration,
        }
    
    def is_failed(self):
        """Verifica se il job è fallito"""
        return self.status == JobStatus.FAILED.value
    
    def is_completed(self):
        """Verifica se il job è completato con successo"""
        return self.status == JobStatus.COMPLETED.value
    
    def is_running(self):
        """Verifica se il job è in esecuzione"""
        return self.status == JobStatus.RUNNING.value
    
    def can_retry(self):
        """Verifica se il job può essere riprovato"""
        return self.retry_count < self.max_retries
