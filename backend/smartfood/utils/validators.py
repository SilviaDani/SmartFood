"""
Validators - Funzioni di validazione dati
"""

import csv
import io
from datetime import datetime
from dateutil import parser as date_parser

def allowed_file(filename):
    """Controlla se il file è un CSV valido"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

def validate_csv_format(file_content):
    """
    Valida il formato del CSV
    Expected columns: school, date, dish_name, portions_prepared, portions_wasted
    
    Returns:
        tuple: (is_valid: bool, result: list[dict] | str)
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
            except (ValueError, AttributeError):
                raise ValueError(f"Row {idx}: portions_prepared must be a valid integer")
            
            # Valida portions_wasted
            try:
                portions_waste_str = normalized_row.get('portions_wasted', '0').strip()
                portions_waste = int(portions_waste_str) if portions_waste_str else 0
                if portions_waste < 0:
                    raise ValueError(f"Row {idx}: portions_wasted cannot be negative")
                if portions_waste > portions_prep:
                    raise ValueError(f"Row {idx}: portions_wasted cannot exceed portions_prepared")
            except (ValueError, AttributeError):
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
