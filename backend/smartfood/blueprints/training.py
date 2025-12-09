"""
Training Blueprint - Endpoints per il training dei modelli AI
"""

from flask import Blueprint, request, jsonify
from smartfood.services import DatasetService, TrainingService
import os

bp = Blueprint('training', __name__, url_prefix='/api')

# Crea istanze dei services
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'uploads')
MODELS_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'trained_models')
dataset_service = DatasetService(UPLOAD_FOLDER)
training_service = TrainingService(MODELS_FOLDER)

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
        return jsonify({
            "success": False,
            "message": f"Error listing datasets: {str(e)}"
        }), 500

@bp.route('/train', methods=['POST'])
def start_training():
    """Avvia il training di un modello in background"""
    try:
        data = request.json
        model_id = data.get('model_id')
        dataset_id = data.get('dataset_id')
        
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
        
        # Avvia il training
        dataset_path = dataset_service.get_dataset_path(dataset_id)
        job_id = training_service.start_training(dataset_path, model_id)
        
        return jsonify({
            "job_id": job_id,
            "status": "started",
            "message": f"Training {model_id} on {dataset_id} started"
        }), 202
    
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error starting training: {str(e)}"
        }), 500

@bp.route('/train/<job_id>/status', methods=['GET'])
def get_training_status(job_id):
    """Controlla lo stato del training"""
    try:
        job = training_service.get_job_status(job_id)
        
        if job is None:
            return jsonify({
                "success": False,
                "message": "Job non trovato"
            }), 404
        
        return jsonify(job), 200
    
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error getting training status: {str(e)}"
        }), 500

@bp.route('/train/<job_id>/results', methods=['GET'])
def get_training_results(job_id):
    """Ottiene i risultati del training"""
    try:
        job = training_service.get_job_status(job_id)
        
        if job is None:
            return jsonify({
                "success": False,
                "message": "Job non trovato"
            }), 404
        
        if job["status"] != "completed":
            return jsonify({
                "success": False,
                "message": "Training non ancora completato"
            }), 400
        
        return jsonify(job["results"]), 200
    
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error getting training results: {str(e)}"
        }), 500
