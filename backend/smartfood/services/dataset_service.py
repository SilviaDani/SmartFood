"""
Dataset Service - Logica per la gestione dei dataset
"""

import os
import csv
from datetime import datetime

class DatasetService:
    """Service per gestire dataset CSV"""
    
    def __init__(self, upload_folder):
        self.upload_folder = upload_folder
        os.makedirs(upload_folder, exist_ok=True)
    
    def save_csv(self, rows):
        """
        Salva i dati del CSV in un file processato
        
        Args:
            rows: lista di dizionari con i dati
            
        Returns:
            tuple: (success: bool, result: int | str)
        """
        try:
            # Crea il file di output con timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(self.upload_folder, f'processed_{timestamp}.csv')
            
            # Salva il CSV elaborato
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                if rows:
                    fieldnames = ['school', 'date', 'dish_name', 'portions_prepared', 'portions_wasted']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
            
            # TODO: Integra con InfluxDB per salvare nel database time-series
            
            return True, len(rows)
        
        except Exception as e:
            return False, str(e)
    
    def list_csv_files(self):
        """
        Lista tutti i file CSV disponibili
        
        Returns:
            list: lista di nomi di file CSV
        """
        try:
            files = []
            if os.path.exists(self.upload_folder):
                for file in os.listdir(self.upload_folder):
                    if file.endswith('.csv'):
                        files.append(file)
            
            return sorted(files, reverse=True)  # Più recenti prima
        
        except Exception as e:
            print(f"Error listing CSV files: {str(e)}")
            return []
    
    def get_dataset_path(self, filename):
        """
        Ottiene il percorso completo di un dataset
        
        Args:
            filename: nome del file
            
        Returns:
            str: percorso completo del file
        """
        return os.path.join(self.upload_folder, filename)
    
    def dataset_exists(self, filename):
        """Controlla se un dataset esiste"""
        path = self.get_dataset_path(filename)
        return os.path.exists(path)
