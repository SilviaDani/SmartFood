"""
Smart Food Backend - API Server
Modulare e scalabile con Blueprints
"""

from flask import Flask
from flask_cors import CORS
from smartfood.blueprints import csv_upload, training, prediction

def create_app():
    """Factory function per creare e configurare l'app Flask"""
    
    app = Flask(__name__)
    
    # Configurazione CORS
    CORS(app, origins=["http://localhost:5173", "http://localhost:3000"])
    
    # Registra i blueprints
    app.register_blueprint(csv_upload.bp)
    app.register_blueprint(training.bp)
    app.register_blueprint(prediction.bp)
    
    # Health check endpoint
    @app.route('/health', methods=['GET'])
    def health_check():
        return {"status": "healthy"}, 200
    
    return app

# Crea l'app
app = create_app()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
