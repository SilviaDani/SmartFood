"""
Smart Food Backend - API Server
Modulare e scalabile con Blueprints e Celery
"""

import os
from flask import Flask
from flask_cors import CORS
from smartfood.models import db
from smartfood.celery_app import celery_app
from smartfood.blueprints import csv_upload, prediction, training_task


def create_app():
    """Factory function per creare e configurare l'app Flask"""
    
    app = Flask(__name__)
    
    # Configurazione database
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv(
        'DATABASE_URL',
        'sqlite:///smartfood.db'
    )
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Inizializza database
    db.init_app(app)
    
    # Configurazione CORS
    CORS(app, origins=["http://localhost:5173", "http://localhost:3000"])
    
    # Registra i blueprints
    app.register_blueprint(csv_upload.bp)
    app.register_blueprint(training_task.bp)
    app.register_blueprint(prediction.bp)
    
    # Health check endpoint
    @app.route('/health', methods=['GET'])
    def health_check():
        return {"status": "healthy"}, 200
    
    # Context per creare le tabelle
    with app.app_context():
        db.create_all()
    
    # Configura Celery
    celery_app.conf.update(app.config)
    
    class ContextTask(celery_app.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)
    
    celery_app.Task = ContextTask
    
    return app


# Crea l'app
app = create_app()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)

