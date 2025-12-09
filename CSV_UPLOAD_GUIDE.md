# CSV Upload Feature - Guida Completa

## Overview
Il componente `CSVUploader` permette agli utenti di caricare file CSV dal frontend. I dati vengono validati e salvati nel backend.

## Struttura

### Frontend
- **Componente React**: `frontend/components/CSVUploader.tsx`
- **Integrazione**: Aggiunto a `DataEntryForm.tsx`
- **API Client**: Usa `frontend/lib/api.ts`

### Backend
- **Endpoint Flask**: `backend/smartfood/csv_uploader.py`
- **Route**: `POST /api/csv/upload`
- **Validazione**: CSV format, dati numerici, date

## Formato CSV Atteso

Il CSV deve contenere esattamente queste colonne (case-insensitive):

```csv
school,date,dish_name,portions_prepared,portions_wasted
Lincoln Elementary,2024-11-10,Spaghetti,120,15
Washington Middle School,2024-11-10,Pasta Primavera,100,8
Roosevelt High School,2024-11-10,Pizza Margherita,80,5
```

### Colonne Richieste
1. **school** (string): Nome della scuola
2. **date** (YYYY-MM-DD): Data nel formato `YYYY-MM-DD`
3. **dish_name** (string): Nome del piatto
4. **portions_prepared** (int): Numero di porzioni preparate (> 0)
5. **portions_wasted** (int): Numero di porzioni scartate (>= 0, <= portions_prepared)

## Validazioni

### Frontend
- Solo file `.csv` sono accettati
- Massimo 10MB per file
- Mostra barra di progresso

### Backend
- Validazione colonne richieste
- Validazione tipo dati (integer, date format)
- Validazione logica (wasted <= prepared, etc.)
- Messaggi di errore dettagliati per ogni riga

## Come Usare

### 1. Preparare il CSV
Crea un file `data.csv`:
```csv
school,date,dish_name,portions_prepared,portions_wasted
Lincoln Elementary,2024-11-10,Spaghetti,120,15
Washington Middle School,2024-11-10,Pasta Primavera,100,8
```

### 2. Caricare dal Frontend
1. Vai alla sezione "Data Entry"
2. Scorri fino a "Bulk Upload CSV Data"
3. Clicca nell'area per selezionare o trascina il file
4. Clicca "Upload CSV"

### 3. Verificare i Risultati
- Il file elaborato viene salvato in `backend/uploads/processed_TIMESTAMP.csv`
- Ricevi una notifica di successo/errore

## Integrazione con InfluxDB

Nel file `backend/smartfood/csv_uploader.py`, cerca il commento `# TODO: Integra con InfluxDB` (linea ~120).

Per salvare i dati in InfluxDB, aggiungi questo codice:

```python
from importer_influxdb import InfluxDBImporter
from datetime import datetime

def save_csv_to_database(rows):
    try:
        importer = InfluxDBImporter()  # Configura con le tue credenziali
        
        for row in rows:
            importer.insert_data(
                school=row['school'],
                date=datetime.strptime(row['date'], '%Y-%m-%d'),
                dish_name=row['dish_name'],
                portions_prepared=int(row['portions_prepared']),
                portions_wasted=int(row['portions_wasted'])
            )
        
        return True, len(rows)
    except Exception as e:
        return False, str(e)
```

## Endpoint API

### POST /api/csv/upload
```bash
curl -X POST http://localhost:8000/api/csv/upload \
  -F "file=@data.csv"
```

**Request**: Form data con file `csv`

**Response Success (200)**:
```json
{
  "success": true,
  "message": "CSV file uploaded and processed successfully",
  "rows_processed": 3
}
```

**Response Error (400/500)**:
```json
{
  "success": false,
  "message": "Row 2: portions_wasted cannot exceed portions_prepared"
}
```

## Struttura delle Cartelle

```
backend/
├── smartfood/
│   ├── csv_uploader.py          ← Nuovo endpoint
│   ├── importer_influxdb.py     ← Da integrare
│   └── ...
└── uploads/                      ← Cartella per file elaborati (creata auto)
    ├── processed_20241110_140530.csv
    └── ...

frontend/
├── components/
│   ├── CSVUploader.tsx          ← Nuovo componente
│   ├── DataEntryForm.tsx        ← Aggiornato
│   └── ...
└── lib/
    └── api.ts
```

## Troubleshooting

### "File must be UTF-8 encoded"
- Salva il CSV con encoding UTF-8 (File → Save As → Encoding: UTF-8 in Excel)

### "Only CSV files are allowed"
- Assicurati che l'estensione del file sia `.csv`

### "Missing required columns"
- Verifica che il CSV contenga esattamente queste colonne: `school, date, dish_name, portions_prepared, portions_wasted`

### CORS Error
- Assicurati che il backend abbia CORS configurato correttamente
- Aggiungi questo al backend:
```python
from flask_cors import CORS
CORS(app)
```

## Prossimi Passi

1. ✅ Implementare il componente frontend
2. ✅ Creare l'endpoint backend
3. ⬜ Integrare con InfluxDB (vedi sezione sopra)
4. ⬜ Aggiungere autenticazione API
5. ⬜ Implementare backup dei file caricati
