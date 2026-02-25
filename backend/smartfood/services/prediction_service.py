"""
Prediction Service - Logica per generare previsioni sui pasti
Usa i dati storici per prevedere le porzioni future
"""

import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import torch
from smartfood.utils.config_loader import get_config_loader
from smartfood.utils.model_registry import get_model_registry
from influxdb_client import InfluxDBClient


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
        
        # Auto-detect CUDA disponibile
        self.device = self._detect_device()
        print(f"[PredictionService] Device detected: {self.device}")
        
        # Cache dei modelli caricati (lazy loading)
        # Struttura: {"model_name_school_name": model_instance}
        self._model_cache = {}
        
        # Context per tracciare la scuola corrente durante la predizione
        self._current_school = None
        
        # Registra automaticamente gli handler di predizione
        # per tutti i modelli disponibili nel YAML
        self._register_prediction_handlers()
    
    def _detect_device(self) -> str:
        """
        Auto-rileva se CUDA è disponibile
        
        Returns:
            "cuda" se disponibile, altrimenti "cpu"
        """
        if torch.cuda.is_available():
            device = "cuda"
            print(f"[PredictionService] CUDA disponibile: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            print("[PredictionService] CUDA non disponibile, usando CPU")
        return device
    
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
    
    def _get_finetuned_model_path(self, school_name: str, model_name: str):
        """
        Controlla se esiste un modello fine-tuned per una scuola
        
        Args:
            school_name: Nome della scuola
            model_name: Nome del modello (chronos, moment, timesfm)
        
        Returns:
            Percorso del modello se esiste, altrimenti None
        """
        # Construisci il percorso: uploads/../trained_models/school_name/model_name_finetuned.pt
        trained_models_dir = os.path.join(
            os.path.dirname(self.uploads_folder),
            'trained_models',
            school_name.lower()
        )
        
        model_file = os.path.join(trained_models_dir, f"{model_name.lower()}_finetuned.pt")
        
        if os.path.exists(model_file):
            print(f"[Models] ✓ Trovato modello fine-tuned: {model_file}")
            return model_file
        else:
            return None
    
    
    def generate_prediction(self, school_name, model_id, start_date, end_date, dish_name=None):
        """
        Genera previsioni per una scuola tra due date (escludendo weekendend)
        
        Args:
            school_name: nome della scuola (es. "scuola1")
            model_id: modello da usare (chronos, moment, timesfm)
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
            # Salva lo school_name come context per i metodi di predizione
            # Così i metodi _predict_with_* possono accedere al nome della scuola
            self._current_school = school_name
            
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
                    "zero_shot": False,
                    "predictions": [],
                    "error": "No working days in the selected date range"
                }
            
            # 1. Carica i dati storici
            df_history = self._load_school_data(school_name, dish_name)
            zero_shot = False
            if df_history is None or len(df_history) == 0:
                # Nessun dato storico disponibile: attiva modalità zero-shot
                # con un contesto sintetico basato su stagionalità e medie italiane
                dish_info = f" (piatto: '{dish_name}')" if dish_name else ""
                print(f"[PredictionService] Nessun dato storico per '{school_name}'{dish_info}, "
                      f"modalità zero-shot attiva")
                zero_shot = True
                df_history = self._create_zero_shot_context(start_date_obj)
            
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
                    "zero_shot": zero_shot,
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
                    "zero_shot": zero_shot,
                    "predictions": [],
                    "error": str(e)
                }
            finally:
                # Pulisci il context
                self._current_school = None
            
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
                "zero_shot": zero_shot,
                "predictions": predictions,
                "error": None
            }
        
        except Exception as e:
            self._current_school = None
            return {
                "school": school_name,
                "model": model_id,
                "start_date": str(start_date) if start_date else "",
                "end_date": str(end_date) if end_date else "",
                "working_days": 0,
                "dish": dish_name,
                "zero_shot": False,
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

    # ==========================================================
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizza e pulisce un DataFrame storico preparandolo per l'inferenza.

        Args:
            df: DataFrame grezzo con almeno le colonne 'date' e 'portions_prepared'

        Returns:
            DataFrame ordinato per data, con tipi corretti e senza NaN
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        if 'portions_prepared' not in df.columns:
            raise ValueError("DataFrame deve avere la colonna 'portions_prepared'")

        df['portions_prepared'] = pd.to_numeric(df['portions_prepared'], errors='coerce')
        df.dropna(subset=['date', 'portions_prepared'], inplace=True)
        df.sort_values('date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df


    # ==========================================================
    def _create_zero_shot_context(self, reference_date, history_days: int = 30) -> pd.DataFrame:
        """
        Crea un contesto sintetico per previsioni zero-shot (nessun dato storico disponibile).

        Genera una serie temporale plausibile per una mensa scolastica italiana:
        - Base: ~150 porzioni/giorno
        - Stagionalità mensile (meno a giugno/agosto, di più in inverno)
        - Piccolo rumore casuale deterministico (seed fisso per riproducibilità)

        Args:
            reference_date: Data di inizio previsione (il contesto finisce il giorno precedente)
            history_days: Numero di giorni lavorativi sintetici da generare (default 30)

        Returns:
            DataFrame con colonne: date, portions_prepared, portions_wasted
        """
        # Fattori stagionali mensili per mense scolastiche italiane
        seasonal_factors = {
            1: 1.05,   # Gennaio
            2: 1.05,   # Febbraio
            3: 1.00,   # Marzo
            4: 0.98,   # Aprile
            5: 0.95,   # Maggio
            6: 0.85,   # Giugno
            7: 0.70,   # Luglio (vacanze estive)
            8: 0.65,   # Agosto (vacanze)
            9: 1.00,   # Settembre (inizio scuola)
            10: 1.05,  # Ottobre
            11: 1.07,  # Novembre
            12: 1.02,  # Dicembre
        }

        BASE_PORTIONS = 150
        rows = []
        rng = np.random.default_rng(seed=42)  # seed fisso per riproducibilità

        # Genera `history_days` giorni lavorativi precedenti alla reference_date
        current = reference_date - timedelta(days=1)
        collected = 0

        while collected < history_days:
            if current.weekday() < 5:  # solo giorni feriali (lun-ven)
                sf = seasonal_factors.get(current.month, 1.0)
                portions = int(BASE_PORTIONS * sf + rng.normal(0, 10))
                portions = max(0, portions)
                wasted = int(portions * rng.uniform(0.08, 0.20))
                rows.append({
                    'date': pd.Timestamp(current),
                    'portions_prepared': portions,
                    'portions_wasted': wasted
                })
                collected += 1
            current -= timedelta(days=1)

        df = pd.DataFrame(rows)
        df.sort_values('date', inplace=True)
        df.reset_index(drop=True, inplace=True)

        print(f"[PredictionService] Contesto zero-shot generato: {len(df)} giorni lavorativi sintetici "
              f"(base {BASE_PORTIONS} porz/giorno, stagionalità mensile)")
        return df

    # =========================================================
    def _fetch_data(self, school_name: str, start_date, end_date, dish_name=None) -> pd.DataFrame:
        """
        Recupera i dati di una scuola da InfluxDB
        
        Args:
            school_name: Nome della scuola (es. "SCUOLA PRIMARIA A ROMA")
            start_date: Data di inizio (str formato YYYY-MM-DD o datetime.date)
            end_date: Data di fine (str formato YYYY-MM-DD o datetime.date)
            dish_name: Nome del piatto/gruppo piatto (opzionale)
            
        Returns:
            DataFrame con le colonne: date, presenze, porzspreco, scuola, [piatto]
        """
        try:
            # Converti le date nel formato RFC3339 richiesto da InfluxDB
            if isinstance(start_date, str):
                start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
            else:
                start_date_obj = start_date
            
            if isinstance(end_date, str):
                end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
            else:
                end_date_obj = end_date
            
            # Aggiungi il giorno successivo a end_date per includere tutte le ore dell'ultimo giorno
            end_date_obj = end_date_obj + timedelta(days=1)
            
            # Converti a RFC3339 format (InfluxDB format)
            start_time_rfc = start_date_obj.strftime('%Y-%m-%dT00:00:00Z')
            end_time_rfc = end_date_obj.strftime('%Y-%m-%dT00:00:00Z')
            
            # Leggi le credenziali da environment variables
            url = os.getenv('INFLUXDB_URL', 'http://localhost:8086')
            token = os.getenv('INFLUXDB_TOKEN', '')
            org = os.getenv('INFLUXDB_ORG', 'smart_food')
            bucket = os.getenv('INFLUXDB_BUCKET', 'smart_food_bucket_2023-2024-2025')
            
            if not token:
                raise ValueError("INFLUXDB_TOKEN not found in environment variables")
            
            # Connettiti a InfluxDB
            print(f"[InfluxDB] Connecting to {url} (org: {org}, bucket: {bucket})")
            client = InfluxDBClient(url=url, token=token, org=org, timeout=30_000)
            query_api = client.query_api()
            
            # Costruisci la query Flux
            # La query recupera sia presenze che porzspreco per la scuola specifica
            query = f'''from(bucket: "{bucket}")
                    |> range(start: {start_time_rfc}, stop: {end_time_rfc})
                    |> filter(fn: (r) => r._measurement == "school_food_waste")
                    |> filter(fn: (r) => r.scuola == "{school_name}")
            '''
            
            # Se specificato un piatto, filtra anche per quello
            if dish_name:
                query += f'''|> filter(fn: (r) => r.gruppopiatto == "{dish_name}")
            '''
            
            # Continua la query: filtra per i field di interesse
            query += f'''|> filter(fn: (r) => r._field == "porzspreco" or r._field == "presenze")
                    |> map(fn: (r) => ({{
                        r with
                        giorno: time(v: int(v: r._time) - int(v: r._time) % 86400000000000)
                    }})
                    |> group(columns: ["giorno", "scuola"'''
            
            if dish_name:
                query += f''', "gruppopiatto"'''
            
            query += f''', "_field"])
                    |> sum()
                    |> pivot(
                        rowKey: ["giorno", "scuola"'''
            
            if dish_name:
                query += f''', "gruppopiatto"'''
            
            query += f'''],
                        columnKey: ["_field"],
                        valueColumn: "_value"
                    )
                    |> yield(name: "result")
            '''
            
            print(f"[InfluxDB] Querying data for school: {school_name} (dish: {dish_name or 'all'})")
            print(f"[InfluxDB] Date range: {start_time_rfc} to {end_time_rfc}")
            
            # Esegui la query
            result = query_api.query(org=org, query=query)
            
            # Trasforma i risultati in una lista di dizionari
            results = []
            for table in result:
                for record in table.records:
                    row = {
                        'date': record.get_time(),
                        'scuola': record.values.get('scuola'),
                        'presenze': record.values.get('presenze'),
                        'porzspreco': record.values.get('porzspreco')
                    }
                    
                    # Aggiungi il piatto se è stato filtrato
                    if dish_name or 'gruppopiatto' in record.values:
                        row['piatto'] = record.values.get('gruppopiatto')
                    
                    results.append(row)
            
            # Trasforma in DataFrame
            if not results:
                print(f"[InfluxDB] ⚠ No data found for school '{school_name}'")
                return pd.DataFrame()
            
            df = pd.DataFrame(results)
            
            # Converti le colonne numeriche
            df['presenze'] = pd.to_numeric(df['presenze'], errors='coerce')
            df['porzspreco'] = pd.to_numeric(df['porzspreco'], errors='coerce')
            
            # Ordina per data
            df.sort_values(by='date', inplace=True)
            df.reset_index(drop=True, inplace=True)
            
            print(f"[InfluxDB] ✓ Retrieved {len(df)} records from InfluxDB")
            client.close()
            
            return df
        
        except Exception as e:
            print(f"[InfluxDB] Error fetching data: {str(e)}")
            raise ValueError(f"Failed to fetch data from InfluxDB: {str(e)}")
    
    # ==========================================================
    # ========================= MOMENT =========================
    # ==========================================================
    def _load_model_moment(self):
        """
        Lazy load del modello MOMENT
        
        Controlla se esiste un modello fine-tuned per la scuola corrente.
        Se esiste, lo carica. Altrimenti carica il modello base zero-shot.
        
        Returns:
            Modello MOMENT (fine-tuned o zero-shot)
        """
        cache_key = f"moment_{self._current_school}" if self._current_school else "moment_base"
        
        if cache_key in self._model_cache:
            print(f"[MOMENT] ✓ Modello caricato da cache: {cache_key}")
            return self._model_cache[cache_key]
        
        finetuned_path = None
        if self._current_school:
            finetuned_path = self._get_finetuned_model_path(self._current_school, "moment")
        
        if finetuned_path:
            try:
                # ==============================================================================
                print(f"[MOMENT] Caricamento modello fine-tuned per {self._current_school}...")
                # ==============================================================================
                from transformers import AutoModel
                model = AutoModel.from_pretrained(
                    finetuned_path,
                    trust_remote_code=True,
                    device_map=self.device
                )
                print(f"[MOMENT] ✓ Modello fine-tuned caricato con successo")
                self._model_cache[cache_key] = model
                return model
            except Exception as e:
                print(f"[MOMENT] ⚠ Errore caricamento fine-tuned ({str(e)}), fallback a zero-shot")
        
        # =======================================================
        print("[MOMENT] Caricamento modello base (zero-shot)...")
        # =======================================================
        """ 
        In MOMENT, la modalità forecasting usa una linear forecasting head inizializzata casualmente, 
        che deve essere fine-tunata prima dell'uso. 
        GitHub Il vero zero-shot in MOMENT funziona per ricostruzione/imputation e anomaly detection, 
        non per il forecasting puro.
        """
        try:
            from momentfm import MOMENTPipeline
        except ImportError:
            raise ValueError(
                "Il pacchetto 'momentfm' non è installato. "
                "MOMENT è temporaneamente disabilitato: richiede transformers==4.33.3, "
                "incompatibile con chronos-forecasting (che richiede transformers>=4.41). "
                "Usa Chronos o TimesFM."
            )

        model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={
                "task_name": "forecasting",
                "forecast_horizon": 90  # converrebbe mettere il massimo di giorni prevedibili dall'utente, ma non c'è un massimo previsto, quindi per ora mettiamo 90 per limitare l'overhead computazionale
            },
        )
        model.init()
        model = model.to(self.device)

        print(f"[MOMENT] ✓ Modello base caricato con successo su {self.device}")
        self._model_cache[cache_key] = model
        return model
    
    # -------------------------------------------------
    def _predict_with_moment(self, df, forecast_days):
        """        
        Args:
            df: DataFrame con date, portions_prepared, portions_wasted
            forecast_days: numero di giorni
            
        Returns:
            list: [{"date": "...", "portions": ..., "confidence": ...}, ...]
        """
        try:
            moment = self._load_model_moment()
            
            portions = df['portions_prepared'].values.astype(np.float32)
            
            if len(portions) < 2:
                raise ValueError("Not enough data points for MOMENT prediction")
            
            # --- Normalizzazione ---
            mean_val = portions.mean()
            std_val = portions.std()
            if np.isnan(std_val) or std_val == 0:
                std_val = mean_val * 0.1 if mean_val != 0 else 1.0
            portions_norm = (portions - mean_val) / std_val

            # --- Padding/troncamento a 512 ---
            SEQ_LEN = 512
            if len(portions_norm) >= SEQ_LEN:
                # Prendi gli ultimi 512 punti
                seq = portions_norm[-SEQ_LEN:]
                mask = np.ones(SEQ_LEN, dtype=np.int64)
            else:
                # Padding a sinistra con zeri
                pad_len = SEQ_LEN - len(portions_norm)
                seq = np.concatenate([np.zeros(pad_len), portions_norm])
                mask = np.concatenate([np.zeros(pad_len, dtype=np.int64),
                                       np.ones(len(portions_norm), dtype=np.int64)])

            # --- Tensori [batch, canali, seq_len] ---
            x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            input_mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0).to(self.device)

            print(f"[MOMENT] Generando previsione per {forecast_days} giorni...")
            moment.eval()
            with torch.no_grad():
                output = moment(x_enc=x, input_mask=input_mask)

            # output.forecast ha shape [1, 1, forecast_horizon]
            # forecast_horizon è fissato al caricamento del modello,
            # quindi prendiamo solo i primi forecast_days valori (o tutti se bastano)
            forecast_norm = output.forecast.squeeze().cpu().numpy()  # shape: (forecast_horizon,)

            # Se il modello ha un orizzonte più corto di forecast_days, avvisa
            if len(forecast_norm) < forecast_days:
                print(f"[MOMENT] ⚠ L'orizzonte del modello ({len(forecast_norm)}) "
                      f"è minore di forecast_days ({forecast_days}). "
                      f"Verranno restituiti solo {len(forecast_norm)} giorni.")
                forecast_days = len(forecast_norm)

            # --- De-normalizzazione ---
            forecast_values = forecast_norm[:forecast_days] * std_val + mean_val

            # --- Costruzione output ---
            last_date = df['date'].max()
            predictions = []

            for day_offset in range(1, forecast_days + 1):
                forecast_date = last_date + timedelta(days=day_offset)
                predicted_portions = int(max(0, round(forecast_values[day_offset - 1])))
                confidence = max(0.45, 0.93 - (day_offset * 0.05))

                predictions.append({
                    "date": forecast_date.strftime('%Y-%m-%d'),
                    "portions": predicted_portions,
                    "confidence": round(confidence, 2)
                })

            print(f"[MOMENT] ✓ Previsione generata con successo")
            return predictions
        
        except Exception as e:
            print(f"[MOMENT] Errore durante la predizione: {str(e)}")
            raise ValueError(f"MOMENT prediction failed: {str(e)}")
    
    # =============================================================
    # ========================= Chronos 2 =========================
    # =============================================================
    def _load_model_chronos(self):
        """
        Lazy load del modello Chronos
        
        Controlla se esiste un modello fine-tuned per la scuola corrente.
        Se esiste, lo carica. Altrimenti carica il modello base zero-shot.
        
        Returns:
            Modello Chronos (fine-tuned o zero-shot)
        """
        # variable to track if a fine-tuned model was loaded (for prediction logic)
        zero_shot_loaded = False

        # Crea una chiave di cache unica basata su school + modello
        cache_key = f"chronos_{self._current_school}" if self._current_school else "chronos_base"
        
        # Se è già in cache, ritornalo (la cache salva la tupla completa)
        if cache_key in self._model_cache:
            print(f"[Chronos] ✓ Modello caricato da cache: {cache_key}")
            return self._model_cache[cache_key]  # già (is_zero_shot, model)
        
        # Controlla se c'è un modello fine-tuned per questa scuola
        finetuned_path = None
        if self._current_school:
            finetuned_path = self._get_finetuned_model_path(self._current_school, "chronos")
        
        if finetuned_path:
            try:
                print(f"[Chronos] Caricamento modello fine-tuned per {self._current_school}...")
                from chronos import Chronos2Pipeline
                model = Chronos2Pipeline.from_pretrained(
                    finetuned_path,
                    device_map=self.device
                )
                print(f"[Chronos] ✓ Modello fine-tuned caricato con successo")
                result = (False, model)
                self._model_cache[cache_key] = result
                return result
            except Exception as e:
                print(f"[Chronos] ⚠ Errore caricamento fine-tuned ({str(e)}), fallback a zero-shot")
                zero_shot_loaded = True
        
        # ========================================================
        print("[Chronos] Caricamento modello base (zero-shot)...")
        # ========================================================
        from chronos import Chronos2Pipeline

        model = Chronos2Pipeline.from_pretrained(
                "amazon/chronos-2", 
                device_map=self.device
            )
        
        print(f"[Chronos] ✓ Modello base caricato con successo su {self.device}")
        result = (zero_shot_loaded, model)
        self._model_cache[cache_key] = result
        return result
    
    def _predict_with_chronos(self, df, forecast_days):
        """
        Genera previsioni usando il modello Chronos (reale)
        
        Args:
            df: DataFrame con date, portions_prepared, portions_wasted
            forecast_days: numero di giorni
            
        Returns:
            list: [{"date": "...", "portions": ..., "confidence": ...}, ...]
        """
        try:
            # Carica il modello (lazy loading)
            is_zero_shot, chronos2 = self._load_model_chronos()
            
            portions = df['portions_prepared'].values.astype(np.float32)
            
            if len(portions) < 2:
                raise ValueError("Not enough data points for Chronos prediction")
            
            # Chronos2Pipeline richiede shape (n_series, n_variates, history_length)
            # unsqueeze(0) → (1, 30), unsqueeze(1) → (1, 1, 30)
            context = torch.tensor(portions, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
            
            print(f"[Chronos] Generando previsione per {forecast_days} giorni...")
            
            # Chronos2Pipeline non accetta num_samples in predict();
            # il numero di campioni è configurato internamente dal modello.
            with torch.no_grad():
                # forecast_samples shape: (1, num_samples, forecast_days)
                # Chronos2Pipeline usa 'inputs=' (rinominato da 'context=' nella v2)
                forecast_samples = chronos2.predict(
                    inputs=context,
                    prediction_length=forecast_days,
                )
            
            # predict() restituisce una lista; [0] = prima serie, [0] = prima variabile
            # → risultante shape (num_samples, forecast_days)
            samples = forecast_samples[0][0].cpu().numpy()
            
            # Stima puntuale con la mediana (più robusta della media agli outlier)
            median_forecast = np.median(samples, axis=0)   # (forecast_days,)
            q10 = np.quantile(samples, 0.10, axis=0)
            q90 = np.quantile(samples, 0.90, axis=0)
            
            last_date = df['date'].max()
            predictions = []
            
            for day_offset in range(1, forecast_days + 1):
                forecast_date = last_date + timedelta(days=day_offset)
                predicted_portions = int(max(0, round(float(median_forecast[day_offset - 1]))))
                
                # Confidence basata sull'ampiezza dell'intervallo predittivo (q90 - q10)
                # Un intervallo stretto = il modello è più sicuro
                interval_width = float(q90[day_offset - 1]) - float(q10[day_offset - 1])
                relative_uncertainty = interval_width / (abs(float(median_forecast[day_offset - 1])) + 1e-6)
                confidence = round(max(0.45, min(0.92, 1.0 - relative_uncertainty * 0.3)), 2)
                
                predictions.append({
                    "date": forecast_date.strftime('%Y-%m-%d'),
                    "portions": predicted_portions,
                    "confidence": confidence
                })
            
            print(f"[Chronos] ✓ Previsione generata con successo")
            return predictions
        
        except Exception as e:
            print(f"[Chronos] Errore durante la predizione: {str(e)}")
            raise ValueError(f"Chronos prediction failed: {str(e)}")
    
    # ===========================================================
    # ========================= TimesFM =========================
    # ===========================================================
    def _load_model_timesfm(self):
        """
        Lazy load del modello TimesFM (Google)
        
        Controlla se esiste un modello fine-tuned per la scuola corrente.
        Se esiste, lo carica. Altrimenti carica il modello base zero-shot.
        
        Returns:
            Modello TimesFM (fine-tuned o zero-shot)
        """
        cache_key = f"timesfm_{self._current_school}" if self._current_school else "timesfm_base"
        
        if cache_key in self._model_cache:
            print(f"[TimesFM] ✓ Modello caricato da cache: {cache_key}")
            return self._model_cache[cache_key]
        
        finetuned_path = None
        if self._current_school:
            finetuned_path = self._get_finetuned_model_path(self._current_school, "timesfm")
        
        if finetuned_path:
            try:
                print(f"[TimesFM] Caricamento modello fine-tuned per {self._current_school}...")
                from transformers import AutoModel
                model = AutoModel.from_pretrained(
                    finetuned_path,
                    trust_remote_code=True,
                    device_map=self.device
                )
                print(f"[TimesFM] ✓ Modello fine-tuned caricato con successo")
                self._model_cache[cache_key] = model
                return model
            except Exception as e:
                print(f"[TimesFM] ⚠ Errore caricamento fine-tuned ({str(e)}), fallback a zero-shot")
        
        # ========================================================
        print("[TimesFM] Caricamento modello base (zero-shot)...")
        # ========================================================
        try:
            import timesfm
        except ImportError:
            raise ValueError(
                "Il pacchetto 'timesfm' non è installato. "
                "TimesFM richiede un ambiente separato: il pacchetto PyPI usa JAX come backend, "
                "incompatibile con l'ambiente PyTorch corrente. "
                "Usa Chronos."
            )
        # Versione 2.0 con backend PyTorch
        tfm = timesfm.TimesFm(
            context_len=512,          # contesto massimo (fino a 2048 per la 2.0)
            horizon_len=64,           # metti il massimo forecast_days che userai
            input_patch_len=32,       # fisso per questo checkpoint
            output_patch_len=128,     # fisso per questo checkpoint
            num_layers=20,            # fisso per questo checkpoint
            model_dims=1280,          # fisso per questo checkpoint
            backend="torch",          # "torch" o "cpu"
        )
        tfm.load_from_checkpoint(repo_id="google/timesfm-2.0-500m-pytorch")
        return tfm
    
    # -------------------------------------------------
    def _predict_with_timesfm(self, df, forecast_days):
        """
        Genera previsioni usando il modello TimesFM di Google (reale)
        
        Args:
            df: DataFrame con date, portions_prepared, portions_wasted
            forecast_days: numero di giorni
            
        Returns:
            list: [{"date": "...", "portions": ..., "confidence": ...}, ...]
        """
        try:
            tfm = self._load_model_timesfm()
            
            portions = df['portions_prepared'].values.astype(np.float32)
            
            if len(portions) < 2:
                raise ValueError("Not enough data points for TimesFM prediction")
            
            print(f"[TimesFM] Generando previsione per {forecast_days} giorni...")

            # TimesFM NON si chiama come una funzione con () e NON accetta tensori PyTorch.
            # Si usa il metodo .forecast() che accetta una lista di array numpy.
            # Non serve torch.no_grad() perché TimesFM gestisce internamente l'inferenza.
            point_forecast, quantile_forecast = tfm.forecast(
                inputs=[portions],  # lista di array numpy, uno per serie
                freq=[0],           # 0 = frequenza giornaliera
            )

            # point_forecast shape: (n_series, horizon_len)
            # quantile_forecast shape: (n_series, horizon_len, n_quantiles)
            forecast_values = point_forecast[0][:forecast_days]  # (forecast_days,)
            q_forecasts = quantile_forecast[0][:forecast_days]   # (forecast_days, n_quantiles)
            q10 = q_forecasts[:, 0]   # quantile 10°
            q90 = q_forecasts[:, -1]  # quantile 90°

            last_date = df['date'].max()
            predictions = []

            for day_offset in range(1, forecast_days + 1):
                forecast_date = last_date + timedelta(days=day_offset)
                predicted_portions = int(max(0, round(forecast_values[day_offset - 1])))

                # Confidence basata sull'intervallo interquantile (q90 - q10)
                interval_width = q90[day_offset - 1] - q10[day_offset - 1]
                relative_uncertainty = interval_width / (abs(forecast_values[day_offset - 1]) + 1e-6)
                confidence = round(max(0.45, min(0.92, 1.0 - relative_uncertainty * 0.3)), 2)

                predictions.append({
                    "date": forecast_date.strftime('%Y-%m-%d'),
                    "portions": predicted_portions,
                    "confidence": confidence
                })

            print(f"[TimesFM] ✓ Previsione generata con successo")
            return predictions
        
        except Exception as e:
            print(f"[TimesFM] Errore durante la predizione: {str(e)}")
            raise ValueError(f"TimesFM prediction failed: {str(e)}")
    
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
