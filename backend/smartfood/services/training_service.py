# """
# Training Service - Logica per il training dei modelli AI
# """

# import os
# import uuid
# from datetime import datetime
# from threading import Thread
# import pandas as pd

# class TrainingService:
#     """Service per gestire il training dei modelli"""
    
#     def __init__(self, models_folder):
#         self.models_folder = models_folder
#         self.jobs = {}  # Salva lo stato dei job (in produzione usa Redis)
#         os.makedirs(models_folder, exist_ok=True)
    
#     def start_training(self, dataset_path, model_id):
#         """
#         Avvia un training in background
        
#         Args:
#             dataset_path: percorso del file CSV
#             model_id: id del modello (moment, chronos)
            
#         Returns:
#             str: job_id
#         """
#         # Crea un ID unico per il job
#         job_id = str(uuid.uuid4())
        
#         # Salva lo stato iniziale
#         self.jobs[job_id] = {
#             "job_id": job_id,
#             "status": "started",
#             "progress": 0,
#             "model": model_id,
#             "dataset": os.path.basename(dataset_path),
#             "error": None,
#             "results": None,
#             "created_at": datetime.now().isoformat()
#         }
        
#         # Avvia il training in background
#         thread = Thread(target=self._train_async, args=(job_id, dataset_path, model_id))
#         thread.daemon = True
#         thread.start()
        
#         return job_id
    
#     def get_job_status(self, job_id):
#         """Ottiene lo stato di un job"""
#         return self.jobs.get(job_id)
    
#     # Non credo serva più con Celery, ma lo lascio per riferimento
#     def _train_async(self, job_id, dataset_path, model_id):
#         """Funzione che gira in background"""
#         try:
#             job = self.jobs[job_id]
            
#             # 1. Carica il dataset
#             job["progress"] = 10
#             df = pd.read_csv(dataset_path)
#             print(f"[Job {job_id}] Loaded dataset with {len(df)} rows")
            
#             # 2. Prepara i dati
#             job["progress"] = 30
#             # TODO: Implementa preprocessing
#             # - Normalizzazione
#             # - Feature engineering
#             # - Handling missing values
            
#             # 3. Addestra il modello
#             job["progress"] = 60
#             if model_id == 'moment':
#                 accuracy = self._train_moment_model(df)
#             elif model_id == 'chronos':
#                 accuracy = self._train_chronos_model(df)
#             else:
#                 raise ValueError(f"Unknown model: {model_id}")
            
#             print(f"[Job {job_id}] Model training completed. Accuracy: {accuracy:.4f}")
            
#             # 4. Salva il modello
#             job["progress"] = 95
#             # TODO: Salva il modello addestrato in self.models_folder
            
#             # 5. Compila i risultati
#             job["progress"] = 100
#             job["status"] = "completed"
#             job["results"] = {
#                 "model": model_id,
#                 "dataset": os.path.basename(dataset_path),
#                 "accuracy": accuracy,
#                 "rows_trained": len(df),
#                 "timestamp": datetime.now().isoformat()
#             }
            
#             print(f"[Job {job_id}] Training completed successfully!")
            
#         except Exception as e:
#             print(f"[Job {job_id}] Training failed: {str(e)}")
#             job = self.jobs.get(job_id)
#             if job:
#                 job["status"] = "failed"
#                 job["error"] = str(e)
    
#     def _train_moment_model(self, df):
#         """
#         Addestra un modello MOMENT
        
#         TODO: Implementa l'addestramento usando la libreria MOMENT
#         """
#         print("[MOMENT] Training started...")
        
#         # Simula l'addestramento per testing
#         import time
#         time.sleep(2)
        
#         # TODO: Sostituisci con vero addestramento
#         accuracy = 0.87
        
#         print(f"[MOMENT] Training completed. Accuracy: {accuracy:.4f}")
#         return accuracy
    
#     def _train_chronos_model(self, df):
#         """
#         Addestra un modello Chronos
        
#         TODO: Implementa l'addestramento usando la libreria Chronos
#         """
#         print("[Chronos] Training started...")
        
#         # Simula l'addestramento per testing
#         import time
#         time.sleep(2)
        
#         # TODO: Sostituisci con vero addestramento
#         accuracy = 0.92
        
#         print(f"[Chronos] Training completed. Accuracy: {accuracy:.4f}")
#         return accuracy
