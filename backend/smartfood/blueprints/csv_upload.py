"""
CSV Upload Blueprint - Endpoints per il caricamento dei CSV
"""

from flask import Blueprint, request, jsonify
from smartfood.utils import allowed_file, validate_csv_format
from smartfood.services import DatasetService
import os

bp = Blueprint('csv_upload', __name__, url_prefix='/api')

# Crea un'istanza del service
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'uploads')
dataset_service = DatasetService(UPLOAD_FOLDER)

@bp.route('/csv/upload', methods=['POST'])
def upload_csv():
    """Endpoint per il caricamento del CSV"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'message': 'No file part in the request'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'message': 'No selected file'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'message': 'Only CSV files are allowed'
            }), 400
        
        # Leggi il contenuto del file
        try:
            content = file.read().decode('utf-8')
        except UnicodeDecodeError:
            return jsonify({
                'success': False,
                'message': 'File must be UTF-8 encoded'
            }), 400
        
        # Valida il formato del CSV
        is_valid, result = validate_csv_format(content)
        if not is_valid:
            return jsonify({
                'success': False,
                'message': f'CSV validation failed: {result}'
            }), 400
        
        rows = result
        
        # Salva nel database
        success, rows_count = dataset_service.save_csv(rows)
        if not success:
            return jsonify({
                'success': False,
                'message': f'Failed to save data: {rows_count}'
            }), 500
        
        return jsonify({
            'success': True,
            'message': f'CSV file uploaded and processed successfully',
            'rows_processed': rows_count
        }), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        }), 500
