# CSV Upload Architecture

## Flusso di Caricamento

```
┌─────────────────────────────────────────────────────────────┐
│ FRONTEND (React + Vite)                                     │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  DataEntryForm Component                            │   │
│  │                                                     │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │ CSVUploader.tsx                             │   │   │
│  │  │                                             │   │   │
│  │  │ • Seleziona file CSV                        │   │   │
│  │  │ • Validazione locale (estensione, size)     │   │   │
│  │  │ • Mostra progress bar                       │   │   │
│  │  │ • Invia FormData → Backend                  │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────┬──────────────────────────────────────────┘
                 │
                 │ HTTP POST
                 │ Content-Type: multipart/form-data
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ BACKEND (Flask/Python)                                      │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  csv_uploader.py → /api/csv/upload                 │   │
│  │                                                     │   │
│  │  1. Validazione file                               │   │
│  │     • Estensione .csv                              │   │
│  │     • UTF-8 encoding                               │   │
│  │     • Max 10MB                                      │   │
│  │                                                     │   │
│  │  2. Parsing CSV                                    │   │
│  │     • Legge colonne richieste                      │   │
│  │     • Validazione formato (YYYY-MM-DD, int)        │   │
│  │     • Validazione logica                           │   │
│  │                                                     │   │
│  │  3. Salvataggio Dati                               │   │
│  │     • File CSV: uploads/processed_*.csv            │   │
│  │     • [TODO] Database: InfluxDB                    │   │
│  │                                                     │   │
│  │  4. Risposta                                       │   │
│  │     • {success: true, rows_processed: N}           │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Componenti Coinvolti

### Frontend Files
```
frontend/
├── components/
│   ├── CSVUploader.tsx        ← Componente upload
│   └── DataEntryForm.tsx       ← Integra CSVUploader
├── lib/
│   └── api.ts                 ← Configurazione API
├── .env                       ← VITE_API_URL=http://localhost:8000
└── .env.production            ← VITE_API_URL=http://backend:8000 (Docker)
```

### Backend Files
```
backend/
├── smartfood/
│   ├── csv_uploader.py        ← Endpoint Flask
│   └── importer_influxdb.py   ← [Integrare per saving]
├── uploads/                   ← [Auto-created] File processati
└── requirements.txt           ← Flask, flask-cors, etc.
```

### Config Files
```
docker-compose.yml              ← Environment per comunicazione
example_data.csv               ← File di test
test_csv_upload.py            ← Script di test
CSV_UPLOAD_GUIDE.md           ← Documentazione dettagliata
```

## Validazioni

### Livello Frontend
✅ Solo `.csv` accettati  
✅ Massimo 10MB  
✅ Feedback visivo (progress bar)  

### Livello Backend
✅ Colonne richieste presenti  
✅ Encoding UTF-8  
✅ Tipo dati corretti (int, date YYYY-MM-DD)  
✅ Validazione logica (wasted ≤ prepared)  
✅ Messaggi di errore per ogni riga  

## Environment Variables

### Sviluppo (localhost)
```env
VITE_API_URL=http://localhost:8000
```

### Docker
```env
VITE_API_URL=http://backend:8000
```

La comunicazione tra frontend e backend avviene automaticamente grazie alla **rete interna di Docker Compose**.

## Response Codes

| Status | Scenario |
|--------|----------|
| 200 | Successo |
| 400 | Validazione fallita (file, formato CSV, dati) |
| 500 | Errore server (saving, encoding) |

## Esempi Request/Response

### Request
```bash
curl -X POST http://localhost:8000/api/csv/upload \
  -F "file=@example_data.csv"
```

### Response - Success
```json
{
  "success": true,
  "message": "CSV file uploaded and processed successfully",
  "rows_processed": 15
}
```

### Response - Error
```json
{
  "success": false,
  "message": "Row 2: portions_wasted cannot exceed portions_prepared"
}
```

## Come Testare

### 1. **Locale (senza Docker)**
```bash
# Terminal 1: Backend
cd backend
python -m smartfood.csv_uploader

# Terminal 2: Frontend
cd frontend
npm run dev

# Terminal 3: Test
python test_csv_upload.py example_data.csv
```

### 2. **Con Docker**
```bash
docker-compose up --build

# Aspetta che i servizi siano pronti, poi:
curl -F "file=@example_data.csv" http://localhost:8000/api/csv/upload
```

## Integrazione InfluxDB

Nel file `backend/smartfood/csv_uploader.py`, la funzione `save_csv_to_database()` ha un placeholder per InfluxDB.

Implementazione:
```python
from importer_influxdb import InfluxDBImporter

def save_csv_to_database(rows):
    try:
        importer = InfluxDBImporter()
        
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

## Roadmap

- [x] Componente frontend CSVUploader
- [x] Validazione client-side
- [x] Endpoint Flask `/api/csv/upload`
- [x] Validazione server-side
- [x] Salvataggio file processato
- [ ] Integrazione InfluxDB
- [ ] Autenticazione API
- [ ] Rate limiting
- [ ] Logging e auditing
- [ ] Batch processing per file grandi
