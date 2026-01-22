"""
Prediction Blueprint - Endpoints per le previsioni dei pasti
"""

from flask import Blueprint, request, jsonify
from smartfood.services import PredictionService
from smartfood.utils.model_registry import get_model_registry
import os

bp = Blueprint('prediction', __name__, url_prefix='/api')

# Crea un'istanza del service
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
prediction_service = PredictionService(UPLOAD_FOLDER)
model_registry = get_model_registry()


@bp.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint per generare previsioni sui pasti
    
    Request body:
    {
        "school": "scuola1",
        "model": "chronos",  # o "moment"
        "start_date": "2025-11-20",  # formato YYYY-MM-DD
        "end_date": "2025-11-28",    # formato YYYY-MM-DD
        "dish_name": "piatto1"  # opzionale
    }
    
    Response:
    {
        "success": bool,
        "school": str,
        "model": str,
        "start_date": str,
        "end_date": str,
        "working_days": int,
        "dish": str or null,
        "predictions": [
            {"date": "2025-11-20", "portions": 120, "confidence": 0.87},
            ...
        ],
        "message": str (se error)
    }
    """
    try:
        data = request.json
        
        # DEBUG: Log what we receive
        print(f"[predict] Received data: {data}")
        
        if not data:
            return jsonify({
                'success': False,
                'message': 'Request body is empty or not JSON'
            }), 400
        
        # Valida i parametri
        school = data.get('school', '').strip() if isinstance(data.get('school'), str) else ''
        model = data.get('model', '').strip().lower() if isinstance(data.get('model'), str) else ''
        start_date = data.get('start_date', '').strip() if isinstance(data.get('start_date'), str) else ''
        end_date = data.get('end_date', '').strip() if isinstance(data.get('end_date'), str) else ''
        dish_name = data.get('dish_name', None)
        if dish_name:
            dish_name = dish_name.strip() if isinstance(dish_name, str) else None
        
        print(f"[predict] Parsed - school='{school}', model='{model}', dates='{start_date}' to '{end_date}', dish='{dish_name}'")
        
        if not school:
            return jsonify({
                'success': False,
                'message': 'school parameter is required and cannot be empty'
            }), 400
        
        # Valida il modello in modo dinamico leggendo il registry
        if not model_registry.is_model_available(model):
            available = ', '.join(model_registry.get_available_models())
            return jsonify({
                'success': False,
                'message': f'model "{model}" is not available. Available models: {available}'
            }), 400
        
        if not start_date or not end_date:
            return jsonify({
                'success': False,
                'message': 'start_date and end_date parameters are required (format: YYYY-MM-DD)'
            }), 400
        
        print(f"[predict] Generating predictions for {school} (dish: {dish_name}) with {model}")
        
        # Genera le previsioni
        result = prediction_service.generate_prediction(school, model, start_date, end_date, dish_name)
        
        if result['error']:
            print(f"[predict] Error: {result['error']}")
            return jsonify({
                'success': False,
                'school': result['school'],
                'model': result['model'],
                'start_date': result['start_date'],
                'end_date': result['end_date'],
                'dish': result['dish'],
                'message': result['error']
            }), 400
        
        print(f"[predict] Success! Generated {len(result['predictions'])} predictions for {result['working_days']} working days")
        
        return jsonify({
            'success': True,
            'school': result['school'],
            'model': result['model'],
            'start_date': result['start_date'],
            'end_date': result['end_date'],
            'working_days': result['working_days'],
            'dish': result['dish'],
            'predictions': result['predictions']
        }), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        }), 500


@bp.route('/dishes/<school>', methods=['GET'])
def get_dishes(school):
    """
    Endpoint per ottenere la lista dei piatti disponibili per una scuola
    
    Response:
    {
        "success": bool,
        "school": str,
        "dishes": ["piatto1", "piatto2", ...]
    }
    """
    try:
        dishes = set()
        
        # Leggi tutti i CSV in uploads
        if os.path.exists(UPLOAD_FOLDER):
            for filename in os.listdir(UPLOAD_FOLDER):
                if not filename.endswith('.csv'):
                    continue
                
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                try:
                    import pandas as pd
                    df = pd.read_csv(filepath)
                    
                    # Filtra per scuola
                    if 'school' in df.columns and 'dish_name' in df.columns:
                        df_school = df[df['school'].str.lower() == school.lower()]
                        
                        if len(df_school) > 0:
                            # Aggiungi i piatti unici di questa scuola
                            dishes.update(df_school['dish_name'].dropna().unique())
                except Exception as e:
                    print(f"[get_dishes] Error reading {filename}: {str(e)}")
                    continue
        
        return jsonify({
            'success': True,
            'school': school,
            'dishes': sorted(list(dishes))
        }), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        }), 500


@bp.route('/schools', methods=['GET'])
def get_schools():
    """
    Endpoint per ottenere la lista delle scuole disponibili
    (basato sui dati storici caricati)
    
    Response:
    {
        "success": bool,
        "schools": ["Lincoln Elementary School", "Washington Middle School", ...]
    }
    """
    try:
        schools = set()
        
        # Leggi tutti i CSV in uploads
        if os.path.exists(UPLOAD_FOLDER):
            for filename in os.listdir(UPLOAD_FOLDER):
                if not filename.endswith('.csv'):
                    continue
                
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                try:
                    import pandas as pd
                    df = pd.read_csv(filepath)
                    
                    if 'school' in df.columns:
                        # Aggiungi tutte le scuole uniche
                        schools.update(df['school'].dropna().unique())
                except Exception as e:
                    print(f"[prediction.py] Error reading {filename}: {str(e)}")
                    continue
        
        return jsonify({
            'success': True,
            'schools': sorted(list(schools))
        }), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        }), 500

@bp.route('/models/available', methods=['GET'])
def get_available_models():
    """
    GET /api/models/available
    
    Ritorna la lista di tutti i modelli disponibili per le predizioni
    
    Response:
    {
        "success": true,
        "models": ["chronos", "moment", "timesfm"],
        "count": 3
    }
    """
    try:
        models = model_registry.get_available_models()
        return jsonify({
            'success': True,
            'models': models,
            'count': len(models)
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        }), 500


@bp.route('/models/detailed', methods=['GET'])
def get_models_detailed():
    """
    GET /api/models/detailed
    
    Ritorna i modelli disponibili con dettagli (display_name, description, ecc.)
    Utile per il frontend per mostrare info complete nel dropdown
    
    Response:
    {
        "success": true,
        "models": [
            {
                "name": "chronos",
                "display_name": "Chronos Forecasting",
                "description": "Advanced time series forecasting...",
                "type": "timeseries",
                "supports_confidence": true
            },
            ...
        ],
        "count": 3
    }
    """
    try:
        all_models = model_registry.config_loader.get_all_models()
        return jsonify({
            'success': True,
            'models': all_models,
            'count': len(all_models)
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        }), 500
