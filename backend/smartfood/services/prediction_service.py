"""
Prediction Service - Logica per generare previsioni sui pasti
Usa i dati storici per prevedere le porzioni future
"""

import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np


def is_weekday(date):
    """Controlla se una data è un giorno feriale (lunedì-venerdì)"""
    return date.weekday() < 5  # 0-4 = lunedì-venerdì, 5-6 = sabato-domenica


def get_working_days(start_date, end_date):
    """
    Ottiene una lista di giorni lavorativi tra due date (inclusive)
    
    Args:
        start_date: datetime.date o str (formato YYYY-MM-DD)
        end_date: datetime.date o str (formato YYYY-MM-DD)
    
    Returns:
        list di datetime.date per i giorni lavorativi
    """
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
    
    working_days = []
    current_date = start_date
    while current_date <= end_date:
        if is_weekday(current_date):
            working_days.append(current_date)
        current_date += timedelta(days=1)
    
    return working_days


class PredictionService:
    """Service per gestire le previsioni dei pasti"""
    
    def __init__(self, uploads_folder):
        """
        Inizializza il service
        
        Args:
            uploads_folder: percorso della cartella uploads con i CSV storici
        """
        self.uploads_folder = uploads_folder
    
    def generate_prediction(self, school_name, model_id, start_date, end_date, dish_name=None):
        """
        Genera previsioni per una scuola tra due date (escludendo weekendend)
        
        Args:
            school_name: nome della scuola (es. "scuola1")
            model_id: modello da usare (moment, chronos)
            start_date: data inizio (str formato YYYY-MM-DD o datetime.date)
            end_date: data fine (str formato YYYY-MM-DD o datetime.date)
            dish_name: nome del piatto (opzionale, None per tutti i piatti)
            
        Returns:
            dict: {
                "school": str,
                "model": str,
                "start_date": str,
                "end_date": str,
                "working_days": int,
                "dish": str or None,
                "predictions": [
                    {"date": "2025-11-13", "portions": 120, "confidence": 0.87},
                    ...
                ],
                "error": None (se successo)
            }
        """
        try:
            # Parsa le date
            if isinstance(start_date, str):
                start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
            else:
                start_date_obj = start_date
                start_date = start_date_obj.strftime('%Y-%m-%d')
            
            if isinstance(end_date, str):
                end_date_obj = datetime.strptime(end_date, '%Y-%m-%d').date()
            else:
                end_date_obj = end_date
                end_date = end_date_obj.strftime('%Y-%m-%d')
            
            # Ottieni i giorni lavorativi
            working_days = get_working_days(start_date_obj, end_date_obj)
            if not working_days:
                return {
                    "school": school_name,
                    "model": model_id,
                    "start_date": start_date,
                    "end_date": end_date,
                    "working_days": 0,
                    "dish": dish_name,
                    "predictions": [],
                    "error": "No working days in the selected date range"
                }
            
            # 1. Carica i dati storici
            df_history = self._load_school_data(school_name, dish_name)
            if df_history is None or len(df_history) == 0:
                dish_filter = f" for dish '{dish_name}'" if dish_name else ""
                return {
                    "school": school_name,
                    "model": model_id,
                    "start_date": start_date,
                    "end_date": end_date,
                    "working_days": len(working_days),
                    "dish": dish_name,
                    "predictions": [],
                    "error": f"No historical data found for school '{school_name}'{dish_filter}"
                }
            
            # 2. Prepara i dati
            df_prepared = self._prepare_data(df_history)
            
            # 3. Genera le previsioni in base al modello
            num_days_to_predict = len(working_days)
            if model_id == 'moment':
                predictions = self._predict_with_moment(df_prepared, num_days_to_predict)
            elif model_id == 'chronos':
                predictions = self._predict_with_chronos(df_prepared, num_days_to_predict)
            else:
                return {
                    "school": school_name,
                    "model": model_id,
                    "start_date": start_date,
                    "end_date": end_date,
                    "working_days": len(working_days),
                    "dish": dish_name,
                    "predictions": [],
                    "error": f"Unknown model: {model_id}"
                }
            
            # 4. Associa le date dei giorni lavorativi alle previsioni
            for i, prediction in enumerate(predictions):
                if i < len(working_days):
                    prediction['date'] = working_days[i].strftime('%Y-%m-%d')
            
            # Se ci sono meno previsioni di giorni lavorativi, tronca
            predictions = predictions[:len(working_days)]
            
            return {
                "school": school_name,
                "model": model_id,
                "start_date": start_date,
                "end_date": end_date,
                "working_days": len(working_days),
                "dish": dish_name,
                "predictions": predictions,
                "error": None
            }
        
        except Exception as e:
            return {
                "school": school_name,
                "model": model_id,
                "start_date": str(start_date) if start_date else "",
                "end_date": str(end_date) if end_date else "",
                "working_days": 0,
                "dish": dish_name,
                "predictions": [],
                "error": str(e)
            }
    
    def _load_school_data(self, school_name, dish_name=None):
        """
        Carica i dati storici di una scuola (e opzionalmente di un piatto specifico)
        da tutti i CSV della cartella uploads
        
        Returns:
            DataFrame con le colonne: date, portions_prepared, portions_wasted
        """
        try:
            all_data = []
            
            # Leggi tutti i file CSV in uploads
            if not os.path.exists(self.uploads_folder):
                return None
            
            for filename in os.listdir(self.uploads_folder):
                if not filename.endswith('.csv'):
                    continue
                
                filepath = os.path.join(self.uploads_folder, filename)
                try:
                    df = pd.read_csv(filepath)
                    
                    # Filtra per scuola
                    if 'school' in df.columns:
                        df_school = df[df['school'].str.lower() == school_name.lower()]
                        
                        # Filtra anche per piatto se specificato
                        if dish_name and 'dish_name' in df_school.columns:
                            df_school = df_school[df_school['dish_name'].str.lower() == dish_name.lower()]
                        
                        if len(df_school) > 0:
                            all_data.append(df_school)
                except Exception as e:
                    print(f"[PredictionService] Error reading {filename}: {str(e)}")
                    continue
            
            if not all_data:
                return None
            
            # Concatena tutti i dati
            df_combined = pd.concat(all_data, ignore_index=True)
            
            # Normalizza i tipi di dato
            df_combined['date'] = pd.to_datetime(df_combined['date'])
            df_combined['portions_prepared'] = pd.to_numeric(df_combined['portions_prepared'], errors='coerce')
            df_combined['portions_wasted'] = pd.to_numeric(df_combined['portions_wasted'], errors='coerce')
            
            # Rimuovi righe con dati mancanti
            df_combined.dropna(subset=['date', 'portions_prepared'], inplace=True)
            
            # Ordina per data
            df_combined.sort_values('date', inplace=True)
            df_combined.reset_index(drop=True, inplace=True)
            
            return df_combined
        
        except Exception as e:
            print(f"[PredictionService] Error loading school data: {str(e)}")
            return None
    
    def _prepare_data(self, df):
        """
        Prepara i dati per il modello
        
        TODO: Implementa preprocessing più sofisticato
        - Normalizzazione
        - Handling di outliers
        - Feature engineering
        """
        # Per ora, usa i dati così come sono
        return df
    
    def _predict_with_moment(self, df, forecast_days):
        """
        Genera previsioni usando il modello MOMENT

        Per ora, simula con una media mobile
        
        Args:
            df: DataFrame con date, portions_prepared, portions_wasted
            forecast_days: numero di giorni
            
        Returns:
            list: [{"date": "...", "portions": ..., "confidence": ...}, ...]

        TODO: Implementa l'integrazione reale con MOMENT
        """
        predictions = []
        
        # Calcola la media mobile (ultimi 14 giorni)
        window = min(14, max(2, len(df)))
        
        # Filtra NaN values
        clean_portions = df['portions_prepared'].dropna()
        if len(clean_portions) == 0:
            raise ValueError("No valid data points found for MOMENT prediction")
        
        avg_portions = clean_portions.tail(window).mean()
        std_portions = clean_portions.tail(window).std()
        
        if np.isnan(std_portions) or std_portions == 0:
            std_portions = avg_portions * 0.1  # 10% della media
        
        # Genera previsioni per i prossimi giorni
        last_date = df['date'].max()
        
        for day_offset in range(1, forecast_days + 1):
            forecast_date = last_date + timedelta(days=day_offset)
            
            # Simula variabilità
            noise = np.random.normal(0, std_portions * 0.1)
            predicted_portions = int(max(0, avg_portions + noise))
            
            # La confidence diminuisce man mano che ci allontaniamo
            confidence = max(0.5, 0.95 - (day_offset * 0.05))
            
            predictions.append({
                "date": forecast_date.strftime('%Y-%m-%d'),
                "portions": predicted_portions,
                "confidence": round(confidence, 2)
            })
        
        return predictions
    
    def _predict_with_chronos(self, df, forecast_days):
        """
        Genera previsioni usando il modello Chronos
        
        Per ora, simula con trend + rumore
        Fallback a media mobile se il trend fallisce
        
        Args:
            df: DataFrame con date, portions_prepared, portions_wasted
            forecast_days: numero di giorni
            
        Returns:
            list: [{"date": "...", "portions": ..., "confidence": ...}, ...]

        TODO: Implementa l'integrazione reale con Chronos
        """
        predictions = []
        
        try:
            # Calcola trend (regressione lineare semplice)
            x = np.arange(len(df))
            y = df['portions_prepared'].values
            
            # Filtra NaN values
            mask = ~np.isnan(y)
            x_clean = x[mask]
            y_clean = y[mask]
            
            if len(x_clean) < 2:
                raise ValueError("Not enough data points for trend calculation")
            
            coeffs = np.polyfit(x_clean, y_clean, 1)  # polinomio di grado 1 (retta)
            trend_slope = coeffs[0]
            
        except Exception as e:
            print(f"[Chronos] Trend calculation failed ({str(e)}), using moving average instead")
            # Fallback a media mobile
            trend_slope = 0
        
        last_date = df['date'].max()
        last_value = df['portions_prepared'].iloc[-1] if not np.isnan(df['portions_prepared'].iloc[-1]) else df['portions_prepared'].mean()
        std_portions = df['portions_prepared'].std()
        
        if np.isnan(std_portions) or std_portions == 0:
            std_portions = last_value * 0.1  # 10% della media
        
        # Genera previsioni con trend
        for day_offset in range(1, forecast_days + 1):
            forecast_date = last_date + timedelta(days=day_offset)
            
            # Applica il trend
            predicted_portions = int(max(0, last_value + (trend_slope * day_offset)))
            
            # Aggiungi rumore
            noise = np.random.normal(0, std_portions * 0.15)
            predicted_portions = int(max(0, predicted_portions + noise))
            
            # La confidence diminuisce man mano che ci allontaniamo
            confidence = max(0.45, 0.92 - (day_offset * 0.06))
            
            predictions.append({
                "date": forecast_date.strftime('%Y-%m-%d'),
                "portions": predicted_portions,
                "confidence": round(confidence, 2)
            })
        
        return predictions
