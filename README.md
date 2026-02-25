# SmartFood

Sistema di previsione delle porzioni per mense scolastiche, basato su modelli AI di time series forecasting (Chronos 2 di Amazon).

---

## Requisiti

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (include Docker Compose)
- ~4 GB di RAM libera (il modello Chronos viene caricato in memoria al primo utilizzo)
- I file Excel storici nella cartella `datas/` (struttura già presente nel repository)

---

## Installazione e avvio

```bash
docker-compose up -d --build
```

Questo comando costruisce e avvia tutti i servizi:

| Servizio        | Porta  | Descrizione                              |
|-----------------|--------|------------------------------------------|
| Frontend        | `3000` | Interfaccia utente React/Vite            |
| Backend API     | `8000` | API Flask + modelli AI                   |
| InfluxDB        | `8086` | Database time series                     |
| Redis           | `6379` | Broker per i task asincroni (Celery)     |
| Celery Worker   | —      | Worker per il training in background     |

Apri il browser su **http://localhost:3000**.

---

## Prima esecuzione

Al primo avvio, la schermata di previsioni mostra:

> *"Importazione dati storici in corso... La prima volta può richiedere alcuni minuti"*

Il sistema sta importando automaticamente i file Excel dalla cartella `datas/` (ultimi 2 anni) in InfluxDB. Attendere il completamento prima di procedere.

---

## Utilizzo

1. Vai alla sezione **Previsioni**
2. Seleziona una **scuola** dall'elenco
3. Scegli un **tipo di piatto** (opzionale — lascia vuoto per tutti i piatti combinati)
4. Seleziona il **modello AI** (attualmente disponibile: Chronos)
5. Imposta il **periodo di previsione** tramite il calendario
6. Clicca **Genera Previsioni**

> Se per la scuola o il piatto selezionato non esistono dati storici, il modello opera in modalità **zero-shot**: genera una stima basata sulla stagionalità media delle mense scolastiche italiane. Il risultato è indicato da un banner giallo.

---

## Comandi utili

```bash
# Avviare (senza rebuild)
docker-compose up -d

# Fermare tutti i servizi
docker-compose down

# Ricostruire dopo modifiche al codice
docker-compose up -d --build

# Vedere i log del backend in tempo reale
docker-compose logs -f backend

# Reimportare tutti i dati da zero (cancella il DB e reimporta)
curl -X POST http://localhost:8000/api/data/reset
```

---

## Struttura del progetto

```
SmartFood/
├── backend/                # API Flask + modelli AI
│   ├── smartfood/
│   │   ├── blueprints/     # Endpoint REST
│   │   ├── services/       # Logica previsioni e training
│   │   └── utils/          # Configurazione modelli
│   ├── config/
│   │   └── models_confidence.yml   # Modelli abilitati
│   └── requirements.txt
├── frontend/               # Interfaccia React/Vite
├── datas/                  # File Excel storici (non modificare)
└── docker-compose.yml
```
