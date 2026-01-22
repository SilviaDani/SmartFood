"""
Prediction Service - Logica per generare previsioni sui pasti
Usa i dati storici per prevedere le porzioni future
"""

import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from smartfood.utils.config_loader import get_config_loader
from smartfood.utils.model_registry import get_model_registry


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
        self.config_loader = get_config_loader()
        self.model_registry = get_model_registry()
        
        # Registra automaticamente gli handler di predizione
        # per tutti i modelli disponibili nel YAML
        self._register_prediction_handlers()
    
    def _register_prediction_handlers(self):
        """
        Registra automaticamente gli handler di predizione per i modelli disponibili.
        
        Cerca metodi named _predict_with_{model_name} per ogni modello nel YAML.
        Se il metodo esiste, lo registra automaticamente nel model_registry.
        
        Questo approccio permette:
        - Aggiungere nuovi modelli al YAML senza modificare questo codice
        - Ogni modello può avere un'implementazione completamente diversa
        - Se un modello non ha un handler, viene semplicemente skippato (errore in fase di runtime)
        """
        available_models = self.model_registry.get_available_models()
        
        for model_name in available_models:
            # Crea il nome del metodo handler
            handler_method_name = f'_predict_with_{model_name.lower()}'
            
            # Verifica se il metodo esiste in questa classe
            if hasattr(self, handler_method_name):
                # Ottieni il metodo
                handler = getattr(self, handler_method_name)
                
                # Registralo nel model_registry
                self.model_registry.register_prediction_handler(model_name, handler)
                print(f"[PredictionService] ✓ Handler registrato per '{model_name}'")
            else:
                print(f"[PredictionService] ⚠ Nessun handler trovato per '{model_name}' (cerca metodo: {handler_method_name})")
    
    
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
            
            # Verifica che il modello sia disponibile
            if not self.model_registry.is_model_available(model_id):
                return {
                    "school": school_name,
                    "model": model_id,
                    "start_date": start_date,
                    "end_date": end_date,
                    "working_days": len(working_days),
                    "dish": dish_name,
                    "predictions": [],
                    "error": f"Unknown model: {model_id}. Available models: {', '.join(self.model_registry.get_available_models())}"
                }
            
            # Chiama il prediction handler registrato nel model_registry
            try:
                predictions = self.model_registry.predict(model_id, df_prepared, num_days_to_predict)
            except ValueError as e:
                return {
                    "school": school_name,
                    "model": model_id,
                    "start_date": start_date,
                    "end_date": end_date,
                    "working_days": len(working_days),
                    "dish": dish_name,
                    "predictions": [],
                    "error": str(e)
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
    
    def _predict_with_timesfm(self, df, forecast_days):
        """
        Genera previsioni usando il modello TimesFM (Google)
        
        Per ora, simula con trend + rumore simile a Chronos
        
        Args:
            df: DataFrame con date, portions_prepared, portions_wasted
            forecast_days: numero di giorni
            
        Returns:
            list: [{"date": "...", "portions": ..., "confidence": ...}, ...]

        TODO: Implementa l'integrazione reale con TimesFM-2.5
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
            
            coeffs = np.polyfit(x_clean, y_clean, 1)
            trend_slope = coeffs[0]
            
        except Exception as e:
            print(f"[TimesFM] Trend calculation failed ({str(e)}), using moving average instead")
            trend_slope = 0
        
        last_date = df['date'].max()
        last_value = df['portions_prepared'].iloc[-1] if not np.isnan(df['portions_prepared'].iloc[-1]) else df['portions_prepared'].mean()
        std_portions = df['portions_prepared'].std()
        
        if np.isnan(std_portions) or std_portions == 0:
            std_portions = last_value * 0.1
        
        # Genera previsioni con trend
        for day_offset in range(1, forecast_days + 1):
            forecast_date = last_date + timedelta(days=day_offset)
            
            # Applica il trend
            predicted_portions = int(max(0, last_value + (trend_slope * day_offset)))
            
            # Aggiungi rumore (leggermente minore di Chronos, poiché TimesFM è più accurato)
            noise = np.random.normal(0, std_portions * 0.12)
            predicted_portions = int(max(0, predicted_portions + noise))
            
            # La confidence è generalmente più alta per TimesFM (zero-shot)
            confidence = max(0.48, 0.94 - (day_offset * 0.05))
            
            predictions.append({
                "date": forecast_date.strftime('%Y-%m-%d'),
                "portions": predicted_portions,
                "confidence": round(confidence, 2)
            })
        
        return predictions
    
    def format_prediction(
        self,
        model_name: str,
        prediction_value: float,
        confidence: float = None
    ) -> dict:
        """
        Formatta la predizione con le informazioni sulla confidenza
        
        Args:
            model_name: Nome del modello (chronos, moment)
            prediction_value: Valore della predizione (porzioni)
            confidence: Valore di confidenza (0-1)
        
        Returns:
            Dict con la predizione formattata per il frontend
        """
        model_config = self.config_loader.get_model_config(model_name)
        vis_config = self.config_loader.get_visualization_config()
        
        result = {
            "model": model_name,
            "model_display_name": model_config.get('display_name') if model_config else model_name,
            "prediction": prediction_value,
            "supports_confidence": self.config_loader.supports_confidence(model_name),
        }
        
        # Aggiungi informazioni sulla confidenza se supportate
        if self.config_loader.supports_confidence(model_name) and confidence is not None:
            result["confidence"] = round(confidence, 2)
            result["confidence_percentage"] = round(confidence * 100, 2)
            result["confidence_level"] = self._get_confidence_level(
                confidence,
                vis_config
            )
            result["confidence_color"] = self._get_confidence_color(
                confidence,
                vis_config
            )
            result["passes_threshold"] = confidence >= self.config_loader.get_confidence_threshold(model_name)
        
        return result
    
    def _get_confidence_level(self, confidence: float, vis_config: dict) -> str:
        """Determina il livello di confidenza (high, medium, low)"""
        high_threshold = vis_config.get('high_threshold', 0.8)
        medium_threshold = vis_config.get('medium_threshold', 0.5)
        
        if confidence >= high_threshold:
            return "high"
        elif confidence >= medium_threshold:
            return "medium"
        else:
            return "low"
    
    def _get_confidence_color(self, confidence: float, vis_config: dict) -> str:
        """Ritorna il colore per la confidenza"""
        level = self._get_confidence_level(confidence, vis_config)
        colors = vis_config.get('confidence_color_scheme', {})
        return colors.get(level, '#999999')
    
    def filter_predictions_by_confidence(
        self,
        predictions: list,
        min_confidence: float = 0.5
    ) -> list:
        """
        Filtra le previsioni in base alla confidenza minima
        
        Args:
            predictions: Lista di predizioni dal metodo generate_prediction
            min_confidence: Soglia minima di confidenza (0-1)
        
        Returns:
            Lista filtrata di previsioni che passano il threshold
        """
        filtered = []
        
        for pred in predictions:
            if 'confidence' in pred:
                if pred.get('confidence', 0) >= min_confidence:
                    filtered.append(pred)
            else:
                # Previsioni senza confidence sono sempre incluse
                filtered.append(pred)
        
        return filtered
