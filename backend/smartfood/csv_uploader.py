"""
CSV Upload API Handler
Gestisce il caricamento e l'elaborazione dei file CSV dal frontend
"""

import os
import csv
import io
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dateutil import parser as date_parser

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "http://localhost:3000"])

# Configurazione
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'csv'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Crea la cartella uploads se non esiste
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Controlla se il file è un CSV valido"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_csv_format(file_content):
    """
    Valida il formato del CSV
    Expected columns: school, date, dish_name, portions_prepared, portions_wasted
    """
    required_columns = {'school', 'date', 'dish_name', 'portions_prepared', 'portions_wasted'}
    
    try:
        # Leggi il CSV
        reader = csv.DictReader(io.StringIO(file_content))
        if reader.fieldnames is None:
            raise ValueError("CSV file is empty")
        
        # Normalizza i nomi delle colonne per il confronto
        actual_columns = {col.strip().lower().replace(' ', '_') for col in reader.fieldnames}
        
        # Controlla che tutte le colonne richieste siano presenti
        if not required_columns.issubset(actual_columns):
            missing = required_columns - actual_columns
            raise ValueError(f"Missing required columns: {', '.join(missing)}")
        
        # Re-leggi per parsare le righe
        reader = csv.DictReader(io.StringIO(file_content))
        rows = []
        
        for idx, row in enumerate(reader, 1):
            # Normalizza i nomi delle colonne in questo dizionario
            normalized_row = {}
            for key, value in row.items():
                normalized_key = key.strip().lower().replace(' ', '_')
                normalized_row[normalized_key] = value
            
            # Validazioni
            if not normalized_row.get('school', '').strip():
                raise ValueError(f"Row {idx}: school cannot be empty")
            
            if not normalized_row.get('date', '').strip():
                raise ValueError(f"Row {idx}: date cannot be empty")
            
            # Valida e normalizza la data (accetta qualsiasi formato)
            try:
                parsed_date = date_parser.parse(normalized_row['date'].strip())
                # Converti in formato YYYY-MM-DD per consistenza
                normalized_row['date'] = parsed_date.strftime('%Y-%m-%d')
            except (ValueError, TypeError):
                raise ValueError(f"Row {idx}: '{normalized_row['date']}' is not a valid date")
            
            if not normalized_row.get('dish_name', '').strip():
                raise ValueError(f"Row {idx}: dish_name cannot be empty")
            
            # Valida portions_prepared
            try:
                portions_prep = int(normalized_row.get('portions_prepared', 0).strip())
                if portions_prep <= 0:
                    raise ValueError(f"Row {idx}: portions_prepared must be > 0")
            except (ValueError, AttributeError) as e:
                raise ValueError(f"Row {idx}: portions_prepared must be a valid integer")
            
            # Valida portions_wasted
            try:
                portions_waste_str = normalized_row.get('portions_wasted', '0').strip()
                portions_waste = int(portions_waste_str) if portions_waste_str else 0
                if portions_waste < 0:
                    raise ValueError(f"Row {idx}: portions_wasted cannot be negative")
                if portions_waste > portions_prep:
                    raise ValueError(f"Row {idx}: portions_wasted cannot exceed portions_prepared")
            except (ValueError, AttributeError) as e:
                raise ValueError(f"Row {idx}: portions_wasted must be a valid integer")
            
            # Aggiungi la riga normalizzata
            rows.append(normalized_row)
        
        if not rows:
            raise ValueError("CSV file is empty (no data rows)")
        
        return True, rows
    
    except csv.Error as e:
        return False, f"CSV parsing error: {str(e)}"
    except ValueError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

def save_csv_to_database(rows):
    """
    Salva i dati del CSV nel database (InfluxDB)
    
    Nota: Implementa in questa funzione la logica per salvare su InfluxDB
    Per ora, salva semplicemente in un file per testing
    """
    try:
        # Crea il file di output con timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(UPLOAD_FOLDER, f'processed_{timestamp}.csv')
        
        # Salva il CSV elaborato
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            if rows:
                fieldnames = ['school', 'date', 'dish_name', 'portions_prepared', 'portions_wasted']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        
        # TODO: Integra con InfluxDB
        # Esempio:
        # from importer_influxdb import InfluxDBImporter
        # importer = InfluxDBImporter()
        # for row in rows:
        #     importer.insert_data(
        #         school=row['school'],
        #         date=row['date'],
        #         dish_name=row['dish_name'],
        #         portions_prepared=int(row['portions_prepared']),
        #         portions_wasted=int(row['portions_wasted'])
        #     )
        
        return True, len(rows)
    
    except Exception as e:
        return False, str(e)

@app.route('/api/csv/upload', methods=['POST'])
def upload_csv():
    """
    Endpoint per il caricamento del CSV
    
    Returns:
        {
            "success": bool,
            "message": str,
            "rows_processed": int (se successful)
        }
    """
    try:
        # Controlla che il file sia presente
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
        
        # Valida l'estensione del file
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
        success, rows_count = save_csv_to_database(rows)
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
