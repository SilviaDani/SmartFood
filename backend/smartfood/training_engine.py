"""
Training API Handler
Gestisce il training dei modelli AI (MOMENT, Chronos, ecc.)
"""

import os
import json
import uuid
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from threading import Thread
from smartfood.utils.model_registry import get_model_registry

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "http://localhost:3000"])

# Cartelle
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
MODELS_FOLDER = os.path.join(os.path.dirname(__file__), 'trained_models')

# Crea cartelle se non esistono
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

# Salva lo stato dei job (in produzione usa Redis o DB)
training_jobs = {}
model_registry = get_model_registry()

@app.route('/api/datasets', methods=['GET'])
def list_datasets():
    """Lista i file CSV disponibili nella cartella uploads"""
    try:
        files = []
        if os.path.exists(UPLOAD_FOLDER):
            for file in os.listdir(UPLOAD_FOLDER):
                if file.endswith('.csv'):
                    files.append(file)
        
        return jsonify({
            "success": True,
            "files": sorted(files, reverse=True)  # Più recenti prima
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error listing datasets: {str(e)}"
        }), 500

@app.route('/api/train', methods=['POST'])
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
        
        # Valida il modello in modo dinamico
        if not model_registry.is_model_available(model_id):
            available = ', '.join(model_registry.get_available_models())
            return jsonify({
                "success": False,
                "message": f"Model '{model_id}' not supported. Available models: {available}"
            }), 400
        
        # Valida il dataset
        dataset_path = os.path.join(UPLOAD_FOLDER, dataset_id)
        if not os.path.exists(dataset_path):
            return jsonify({
                "success": False,
                "message": f"Dataset '{dataset_id}' not found"
            }), 404
        
        # Crea un ID unico per il job
        job_id = str(uuid.uuid4())
        
        # Salva lo stato iniziale
        training_jobs[job_id] = {
            "job_id": job_id,
            "status": "started",
            "progress": 0,
            "model": model_id,
            "dataset_id": dataset_id,
            "error": None,
            "results": None,
            "created_at": datetime.now().isoformat()
        }
        
        # Avvia il training in background
        thread = Thread(target=train_model_async, args=(job_id, dataset_id, model_id))
        thread.daemon = True
        thread.start()
        
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

@app.route('/api/train/<job_id>/status', methods=['GET'])
def get_training_status(job_id):
    """Controlla lo stato del training"""
    if job_id not in training_jobs:
        return jsonify({
            "success": False,
            "message": "Job non trovato"
        }), 404
    
    return jsonify(training_jobs[job_id]), 200

@app.route('/api/train/<job_id>/results', methods=['GET'])
def get_training_results(job_id):
    """Ottiene i risultati del training"""
    if job_id not in training_jobs:
        return jsonify({
            "success": False,
            "message": "Job non trovato"
        }), 404
    
    job = training_jobs[job_id]
    
    if job["status"] != "completed":
        return jsonify({
            "success": False,
            "message": "Training non ancora completato"
        }), 400
    
    return jsonify(job["results"]), 200

def train_model_async(job_id, dataset_id, model_id):
    """
    Funzione che gira in background
    Allena il modello e aggiorna lo stato del job
    """
    try:
        job = training_jobs[job_id]
        
        # 1. Carica il dataset
        job["progress"] = 10
        dataset_path = os.path.join(UPLOAD_FOLDER, dataset_id)
        
        import pandas as pd
        df = pd.read_csv(dataset_path)
        print(f"[Job {job_id}] Loaded dataset with {len(df)} rows")
        
        # 2. Prepara i dati
        job["progress"] = 30
        # TODO: Implementa il preprocessing dei dati
        # - Normalizzazione
        # - Feature engineering
        # - Handling missing values
        
        # 3. Addestra il modello
        job["progress"] = 60
        if model_id == 'moment':
            accuracy = train_moment_model(df)
        elif model_id == 'chronos':
            accuracy = train_chronos_model(df)
        else:
            raise ValueError(f"Unknown model: {model_id}")
        
        print(f"[Job {job_id}] Model training completed. Accuracy: {accuracy:.4f}")
        
        # 4. Valuta il modello
        job["progress"] = 85
        # TODO: Implementa la valutazione
        
        # 5. Salva il modello
        job["progress"] = 95
        model_save_path = os.path.join(MODELS_FOLDER, f"{model_id}_{job_id}.pkl")
        # TODO: Salva il modello addestrato
        
        # 6. Compila i risultati
        job["progress"] = 100
        job["status"] = "completed"
        job["results"] = {
            "model": model_id,
            "dataset": dataset_id,
            "accuracy": accuracy,
            "rows_trained": len(df),
            "model_path": model_save_path,
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"[Job {job_id}] Training completed successfully!")
        
    except Exception as e:
        print(f"[Job {job_id}] Training failed: {str(e)}")
        job = training_jobs.get(job_id)
        if job:
            job["status"] = "failed"
            job["error"] = str(e)

def train_moment_model(df):
    """
    Addestra un modello MOMENT
    
    TODO: Implementa l'addestramento usando la libreria MOMENT
    """
    print("[MOMENT] Training started...")
    
    # Simula l'addestramento per testing
    import time
    time.sleep(3)
    
    # TODO: Sostituisci con vero addestramento
    accuracy = 0.87
    
    print(f"[MOMENT] Training completed. Accuracy: {accuracy:.4f}")
    return accuracy

def train_chronos_model(df):
    """
    Addestra un modello Chronos
    
    TODO: Implementa l'addestramento usando la libreria Chronos
    """
    print("[Chronos] Training started...")
    
    # Simula l'addestramento per testing
    import time
    time.sleep(3)
    
    # TODO: Sostituisci con vero addestramento
    accuracy = 0.92
    
    print(f"[Chronos] Training completed. Accuracy: {accuracy:.4f}")
    return accuracy

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
