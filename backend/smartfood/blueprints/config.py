"""
Config API Blueprint - Espone la configurazione dei modelli al frontend
"""

from flask import Blueprint, jsonify
from smartfood.utils.config_loader import get_config_loader

config_bp = Blueprint('config', __name__, url_prefix='/api/config')


@config_bp.route('/models', methods=['GET'])
def get_models_config():
    """
    GET /api/config/models
    
    Ritorna la configurazione completa dei modelli e visualizzazione
    
    Response:
    {
        "models": [
            {
                "name": "chronos",
                "display_name": "Chronos Forecasting",
                "type": "timeseries",
                "supports_confidence": true,
                ...
            }
        ],
        "visualization": {
            "confidence_color_scheme": {...},
            "high_threshold": 0.80,
            ...
        }
    }
    """
    try:
        config_loader = get_config_loader()
        return jsonify({
            'success': True,
            'models': config_loader.get_all_models(),
            'visualization': config_loader.get_visualization_config()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@config_bp.route('/models/with-confidence', methods=['GET'])
def get_models_with_confidence():
    """
    GET /api/config/models/with-confidence
    
    Ritorna solo i modelli che supportano confidence scores
    
    Response:
    {
        "models": ["chronos", "moment"]
    }
    """
    try:
        config_loader = get_config_loader()
        return jsonify({
            'success': True,
            'models': config_loader.get_models_with_confidence()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@config_bp.route('/models/without-confidence', methods=['GET'])
def get_models_without_confidence():
    """
    GET /api/config/models/without-confidence
    
    Ritorna solo i modelli senza confidence scores
    
    Response:
    {
        "models": []
    }
    """
    try:
        config_loader = get_config_loader()
        return jsonify({
            'success': True,
            'models': config_loader.get_models_without_confidence()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@config_bp.route('/models/<model_name>', methods=['GET'])
def get_model_config(model_name):
    """
    GET /api/config/models/{model_name}
    
    Ritorna la configurazione di un modello specifico
    
    Response:
    {
        "success": true,
        "model": {
            "name": "chronos",
            "display_name": "Chronos Forecasting",
            ...
        }
    }
    """
    try:
        config_loader = get_config_loader()
        model_config = config_loader.get_model_config(model_name)
        
        if not model_config:
            return jsonify({
                'success': False,
                'error': f"Model '{model_name}' not found"
            }), 404
        
        return jsonify({
            'success': True,
            'model': model_config
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
