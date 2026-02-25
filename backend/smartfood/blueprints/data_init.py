"""
Data Init Blueprint - Importazione iniziale dei dati storici da Excel a InfluxDB

Espone un solo endpoint:
  POST /api/data/initialize
    - Controlla se InfluxDB contiene già dati
    - Se vuoto, importa tutti i file reporttipo-1-dettagliato.xlsx dalla cartella datas/
    - Idempotente: chiamabile più volte senza duplicare i dati
"""

import os
import glob
import pandas as pd
from flask import Blueprint, jsonify
from influxdb_client import InfluxDBClient, Point, WritePrecision, BucketRetentionRules
from influxdb_client.client.write_api import SYNCHRONOUS
from smartfood.exporter_influxdb import clean_school_name, add_macrocategoria_column

bp = Blueprint('data_init', __name__, url_prefix='/api/data')

# Path alla cartella datas montata in Docker
DATAS_FOLDER = os.getenv('DATAS_FOLDER', '/app/datas')
MACROCATEGORIE_FILE = os.path.join(DATAS_FOLDER, 'Piatti_Categorizzati.xlsx')

INFLUXDB_URL   = lambda: os.getenv('INFLUXDB_URL',    'http://localhost:8086')
INFLUXDB_TOKEN = lambda: os.getenv('INFLUXDB_TOKEN',  '')
INFLUXDB_ORG   = lambda: os.getenv('INFLUXDB_ORG',    'smart_food')
INFLUXDB_BUCKET= lambda: os.getenv('INFLUXDB_BUCKET', 'smart_food_bucket_2023-2024-2025')


def _influxdb_has_data() -> bool:
    """
    Controlla se il bucket InfluxDB contiene dati VALIDI:
    verifica che esistano tag 'scuola' non vuoti.
    Usa schema.tagValues (indexed) per evitare full scan timeout.
    """
    client = InfluxDBClient(
        url=INFLUXDB_URL(), token=INFLUXDB_TOKEN(),
        org=INFLUXDB_ORG(), timeout=15_000
    )
    try:
        query_api = client.query_api()
        query = f'''
            import "influxdata/influxdb/schema"
            schema.tagValues(
              bucket: "{INFLUXDB_BUCKET()}",
              tag: "scuola",
              predicate: (r) => r._measurement == "school_food_waste",
              start: -8y
            )
            |> limit(n: 1)
        '''
        result = query_api.query(org=INFLUXDB_ORG(), query=query)
        schools = [r.get_value() for table in result for r in table.records if r.get_value()]
        print(f"[DataInit] Check dati esistenti: trovate {len(schools)} scuole: {schools[:3]}")
        return len(schools) > 0
    except Exception as e:
        print(f"[DataInit] Errore nel check dati: {e}")
        return False
    finally:
        client.close()


def _ensure_bucket_exists(client: InfluxDBClient):
    """Crea il bucket se non esiste ancora. Non lo cancella mai."""
    buckets_api = client.buckets_api()
    existing = buckets_api.find_buckets().buckets
    if not any(b.name == INFLUXDB_BUCKET() for b in existing):
        retention = BucketRetentionRules(type="expire", every_seconds=0)
        buckets_api.create_bucket(
            bucket_name=INFLUXDB_BUCKET(),
            org=INFLUXDB_ORG(),
            retention_rules=[retention]
        )
        print(f"[DataInit] Bucket '{INFLUXDB_BUCKET()}' creato.")
    else:
        print(f"[DataInit] Bucket '{INFLUXDB_BUCKET()}' già esistente.")


def _import_excel_file(path: str, write_api, imported: list, errors: list):
    """
    Importa un singolo file Excel in InfluxDB senza cancellare il bucket.
    Stessa logica di excel2influxdb() ma riusa il write_api esistente.
    """
    try:
        df = pd.read_excel(path, engine='openpyxl')
        df.columns = (df.columns.str.strip().str.lower()
                        .str.replace(" ", "_").str.replace("-", "_"))

        # Parsing robusto delle date: gestisce formati con o senza orario
        # es. "01/01/2019", "01/01/2019 00:00:00", "01/01/2019 00:00"
        df['data'] = pd.to_datetime(df['data'], dayfirst=True, format='mixed')
        df = df.drop_duplicates()

        # Normalizza colonne opzionali mancanti nei file più vecchi
        if 'gruppopiatto' not in df.columns:
            df['gruppopiatto'] = 'N/A'
        if 'ragionesociale' not in df.columns:
            df['ragionesociale'] = 'N/A'

        if os.path.exists(MACROCATEGORIE_FILE):
            df = add_macrocategoria_column(df, MACROCATEGORIE_FILE)
        else:
            df['macrocategoria'] = 'N/A'

        grouped = df.groupby(
            ['data', 'scuola', 'ragionesociale', 'gruppopiatto', 'piatto', 'macrocategoria']
        ).agg({'presenze': 'sum', 'porzspreco': 'sum'}).reset_index()

        batch = []
        for _, row in grouped.iterrows():
            scuola_cleaned = clean_school_name(row)
            point = (
                Point("school_food_waste")
                .tag("ragionesociale", str(row["ragionesociale"]))
                .tag("scuola", scuola_cleaned)
                .tag("gruppopiatto", str(row["gruppopiatto"]))
                .tag("piatto", str(row["piatto"]))
                .tag("macrocategoria", str(row.get("macrocategoria", "N/A")))
                .field("presenze", float(row["presenze"]))
                .field("porzspreco", float(row["porzspreco"]))
                .time(row["data"], WritePrecision.S)
            )
            batch.append(point)

        write_api.write(
            bucket=INFLUXDB_BUCKET(), org=INFLUXDB_ORG(), record=batch
        )
        print(f"[DataInit] ✓ Importato: {path} ({len(batch)} righe)")
        imported.append({'file': path, 'rows': len(batch)})

    except Exception as e:
        print(f"[DataInit] ✗ Errore su {path}: {e}")
        errors.append({'file': path, 'error': str(e)})


def _find_excel_files(years: int = 2) -> list:
    """
    Trova i file reporttipo-1-dettagliato.xlsx in datas/
    limitandosi agli ultimi `years` anni e scartando le cartelle _old.
    """
    from datetime import date
    current_year = date.today().year
    allowed_years = {str(current_year - i) for i in range(years)}

    pattern = os.path.join(DATAS_FOLDER, '**', 'reporttipo-1-dettagliato.xlsx')
    all_files = glob.glob(pattern, recursive=True)

    filtered = []
    for f in all_files:
        normalized = f.replace('\\', '/')
        if '_old' in normalized:
            continue
        # Includi solo i file la cui cartella contiene uno degli anni ammessi
        if any(yr in normalized for yr in allowed_years):
            filtered.append(f)

    print(f"[DataInit] File trovati (ultimi {years} anni): {len(filtered)} su {len(all_files)} totali")
    return filtered


@bp.route('/reset', methods=['POST'])
def reset_data():
    """
    Elimina tutti i dati dal bucket (drop + recreate) e forza un re-import completo.
    Molto più veloce del delete record per record, anche con grandi dataset.
    """
    try:
        client = InfluxDBClient(
            url=INFLUXDB_URL(), token=INFLUXDB_TOKEN(),
            org=INFLUXDB_ORG(), timeout=60_000
        )
        try:
            buckets_api = client.buckets_api()
            # Trova e elimina il bucket
            existing = buckets_api.find_buckets().buckets
            bucket = next((b for b in existing if b.name == INFLUXDB_BUCKET()), None)
            if bucket:
                buckets_api.delete_bucket(bucket)
                print(f"[DataInit] Bucket '{INFLUXDB_BUCKET()}' eliminato.")
        finally:
            client.close()

        # Ricrea il bucket e reimporta tutto
        return initialize_data(force=True)

    except Exception as e:
        print(f"[DataInit] Errore durante reset: {e}")
        return jsonify({
            'success': False,
            'already_initialized': False,
            'imported': [],
            'errors': [],
            'message': f'Errore durante reset: {str(e)}'
        }), 500


@bp.route('/initialize', methods=['POST'])
def initialize_data(force: bool = False):
    """
    Controlla se InfluxDB è vuoto. Se sì, importa tutti i file Excel storici.
    Idempotente: se i dati esistono già non fa nulla.

    Response:
    {
        "success": bool,
        "already_initialized": bool,
        "imported": [{"file": "...", "rows": N}, ...],
        "errors": [{"file": "...", "error": "..."}, ...],
        "message": str
    }
    """
    try:
        # Controlla se ci sono già dati validi (con tag scuola)
        if not force and _influxdb_has_data():
            return jsonify({
                'success': True,
                'already_initialized': True,
                'imported': [],
                'errors': [],
                'message': 'Database già inizializzato.'
            }), 200

        # Trova tutti i file Excel da importare
        excel_files = _find_excel_files()

        if not excel_files:
            return jsonify({
                'success': False,
                'already_initialized': False,
                'imported': [],
                'errors': [],
                'message': f'Nessun file Excel trovato in {DATAS_FOLDER}'
            }), 404

        print(f"[DataInit] Trovati {len(excel_files)} file Excel da importare.")

        # Crea client InfluxDB e importa tutto
        client = InfluxDBClient(
            url=INFLUXDB_URL(), token=INFLUXDB_TOKEN(),
            org=INFLUXDB_ORG(), timeout=600_000  # 10 min per import grandi
        )

        try:
            _ensure_bucket_exists(client)
            # SYNCHRONOUS garantisce che i dati siano indicizzati prima che il client risponda
            write_api = client.write_api(write_options=SYNCHRONOUS)

            imported = []
            errors = []

            for excel_path in sorted(excel_files):
                _import_excel_file(excel_path, write_api, imported, errors)

            # Con SYNCHRONOUS non serve flush() esplicito, ma chiudiamo comunque il client
            write_api.close()

        finally:
            client.close()

        message = (
            f'Importati {len(imported)} file su {len(excel_files)}.'
            + (f' {len(errors)} errori.' if errors else '')
        )
        print(f"[DataInit] {message}")

        return jsonify({
            'success': len(imported) > 0,
            'already_initialized': False,
            'imported': imported,
            'errors': errors,
            'message': message
        }), 200

    except Exception as e:
        print(f"[DataInit] Errore generale: {e}")
        return jsonify({
            'success': False,
            'already_initialized': False,
            'imported': [],
            'errors': [],
            'message': f'Errore: {str(e)}'
        }), 500
