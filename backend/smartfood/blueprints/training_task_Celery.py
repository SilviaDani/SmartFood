"""
Training Blueprint - Endpoints per il training dei modelli AI con Celery
"""

import os
import uuid
import logging
from datetime import datetime
from flask import Blueprint, request, jsonify
from smartfood.models import TrainingJob, JobStatus, db
from smartfood.services import DatasetService
from smartfood.tasks.training_task import train_model

logger = logging.getLogger(__name__)

bp = Blueprint('training', __name__, url_prefix='/api')

# Crea istanze dei services
UPLOAD_FOLDER = '/app/uploads'
MODELS_FOLDER = '/app/trained_models'
dataset_service = DatasetService(UPLOAD_FOLDER)


@bp.route('/datasets', methods=['GET'])
def list_datasets():
    """Lista i file CSV disponibili nella cartella uploads"""
    try:
        files = dataset_service.list_csv_files()
        return jsonify({
            "success": True,
            "files": files
        }), 200
    except Exception as e:
        logger.error(f"Error listing datasets: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Error listing datasets: {str(e)}"
        }), 500


@bp.route('/train', methods=['POST'])
def start_training():
    """
    Avvia il training di un modello in coda con Celery
    
    Body:
    {
        "model_id": "chronos" | "moment",
        "dataset_id": "filename.csv",
        "priority": 0,  # opzionale, default 0
        "scheduled_at": "2025-01-15T10:30:00"  # opzionale per scheduling
    }
    """
    try:
        data = request.json or {}
        model_id = data.get('model_id')
        dataset_id = data.get('dataset_id')
        priority = data.get('priority', 0)
        scheduled_at = data.get('scheduled_at')
        
        # Validazione input
        if not model_id or not dataset_id:
            return jsonify({
                "success": False,
                "message": "model_id e dataset_id sono obbligatori"
            }), 400
        
        # Valida il modello
        if model_id not in ['moment', 'chronos']:
            return jsonify({
                "success": False,
                "message": f"Model '{model_id}' not supported. Use: moment, chronos"
            }), 400
        
        # Valida il dataset
        if not dataset_service.dataset_exists(dataset_id):
            return jsonify({
                "success": False,
                "message": f"Dataset '{dataset_id}' not found"
            }), 404
        
        # Crea il job nel database
        job_id = str(uuid.uuid4())
        dataset_path = dataset_service.get_dataset_path(dataset_id)
        
        job = TrainingJob(
            id=job_id,
            model_id=model_id,
            dataset_id=dataset_id,
            status=JobStatus.PENDING.value,
            priority=priority,
            scheduled_at=datetime.fromisoformat(scheduled_at) if scheduled_at else None,
            max_retries=5
        )
        
        db.session.add(job)
        db.session.commit()
        
        # Avvia il task Celery
        # Se è schedulato per dopo, potremmo usare apply_async con eta
        eta = None
        if scheduled_at:
            eta = datetime.fromisoformat(scheduled_at)
        
        celery_task = train_model.apply_async(
            args=[job_id, dataset_path, model_id],
            task_id=f"train-{job_id}",
            eta=eta,
            priority=10 - priority  # Celery usa numeri più alti = priorità più alta
        )
        
        # Salva il celery task ID
        job.celery_task_id = celery_task.id
        db.session.commit()
        
        logger.info(f"Training job created: {job_id} - model={model_id}, dataset={dataset_id}")
        
        return jsonify({
            "job_id": job_id,
            "celery_task_id": celery_task.id,
            "status": "pending",
            "message": f"Training {model_id} on {dataset_id} queued for execution"
        }), 202
    
    except Exception as e:
        logger.error(f"Error starting training: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "message": f"Error starting training: {str(e)}"
        }), 500


@bp.route('/train/<job_id>/status', methods=['GET'])
def get_training_status(job_id):
    """
    Ottiene lo stato di un job di training
    
    Response:
    {
        "id": "...",
        "status": "pending|running|completed|failed",
        "progress": 0-100,
        "current_step": "...",
        "accuracy": 0.92,
        "...": "..."
    }
    """
    try:
        job = TrainingJob.query.get(job_id)
        
        if job is None:
            return jsonify({
                "success": False,
                "message": "Job non trovato"
            }), 404
        
        return jsonify(job.to_dict()), 200
    
    except Exception as e:
        logger.error(f"Error getting training status: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Error getting training status: {str(e)}"
        }), 500


@bp.route('/train/<job_id>/results', methods=['GET'])
def get_training_results(job_id):
    """Ottiene i risultati del training completato"""
    try:
        job = TrainingJob.query.get(job_id)
        
        if job is None:
            return jsonify({
                "success": False,
                "message": "Job non trovato"
            }), 404
        
        if not job.is_completed():
            return jsonify({
                "success": False,
                "message": "Training non ancora completato",
                "status": job.status,
                "progress": job.progress
            }), 400
        
        return jsonify({
            "job_id": job.id,
            "model_id": job.model_id,
            "dataset_id": job.dataset_id,
            "status": job.status,
            "accuracy": job.accuracy,
            "loss": job.loss,
            "metrics": job.metrics,
            "model_path": job.model_path,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "duration": (job.completed_at - job.started_at).total_seconds() if job.completed_at and job.started_at else None,
        }), 200
    
    except Exception as e:
        logger.error(f"Error getting training results: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Error getting training results: {str(e)}"
        }), 500


@bp.route('/train/<job_id>/cancel', methods=['POST'])
def cancel_training(job_id):
    """Cancella un job di training se non è ancora iniziato"""
    try:
        job = TrainingJob.query.get(job_id)
        
        if job is None:
            return jsonify({
                "success": False,
                "message": "Job non trovato"
            }), 404
        
        if job.status == JobStatus.PENDING.value:
            # Revoca il task Celery se non è ancora iniziato
            if job.celery_task_id:
                from smartfood.celery_app import celery_app
                celery_app.control.revoke(job.celery_task_id, terminate=True)
            
            job.status = JobStatus.CANCELLED.value
            job.completed_at = datetime.utcnow()
            db.session.commit()
            
            logger.info(f"Training job cancelled: {job_id}")
            
            return jsonify({
                "success": True,
                "message": "Training job cancelled",
                "job_id": job_id
            }), 200
        
        elif job.status == JobStatus.RUNNING.value:
            # Revoca il task in esecuzione
            if job.celery_task_id:
                from smartfood.celery_app import celery_app
                celery_app.control.revoke(job.celery_task_id, terminate=True)
            
            job.status = JobStatus.CANCELLED.value
            job.completed_at = datetime.utcnow()
            db.session.commit()
            
            logger.info(f"Training job terminated: {job_id}")
            
            return jsonify({
                "success": True,
                "message": "Training job terminated",
                "job_id": job_id
            }), 200
        
        else:
            return jsonify({
                "success": False,
                "message": f"Cannot cancel job with status: {job.status}"
            }), 400
    
    except Exception as e:
        logger.error(f"Error cancelling training: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Error cancelling training: {str(e)}"
        }), 500


@bp.route('/train/history', methods=['GET'])
def training_history():
    """
    Ottiene lo storico dei job di training
    
    Query params:
    - limit: numero di risultati (default 20)
    - offset: offset dei risultati (default 0)
    - status: filtra per stato (pending, running, completed, failed)
    - model_id: filtra per modello
    """
    try:
        limit = request.args.get('limit', 20, type=int)
        offset = request.args.get('offset', 0, type=int)
        status = request.args.get('status')
        model_id = request.args.get('model_id')
        
        query = TrainingJob.query
        
        if status:
            query = query.filter_by(status=status)
        if model_id:
            query = query.filter_by(model_id=model_id)
        
        total = query.count()
        jobs = query.order_by(TrainingJob.created_at.desc()).limit(limit).offset(offset).all()
        
        return jsonify({
            "total": total,
            "limit": limit,
            "offset": offset,
            "jobs": [job.to_dict() for job in jobs]
        }), 200
    
    except Exception as e:
        logger.error(f"Error getting training history: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Error getting training history: {str(e)}"
        }), 500

