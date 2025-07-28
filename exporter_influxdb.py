import pandas as pd
from influxdb_client import InfluxDBClient, Point, WritePrecision, BucketRetentionRules
from datetime import datetime, timezone
import argparse
import re
import unicodedata
import string

def clean_school_name(row):
    # Normalize and remove accents
    scuola = str(row["scuola"]).strip()
    scuola = unicodedata.normalize('NFD', scuola)
    scuola = ''.join(c for c in scuola if unicodedata.category(c) != 'Mn')
    scuola = scuola.upper()

    # Remove contents in parentheses BEFORE punctuation cleanup
    scuola = re.sub(r'\(.*?\)', '', scuola)

    # Replace L.L. and ":" with space-friendly characters + FIX typo error SEGNOMIGNO
    scuola = scuola.replace("L.L.", "L.").replace(":", ".").replace("SEGNOMIGNO", "SEGROMIGNO")

    # Remove trailing numbers
    scuola = re.sub(r'\d.*$', '', scuola)

    # Remove specific suffixes
    scuola = re.sub(r'\s+(GIU|SU)\b', '', scuola)

    # Replace punctuation with space, collapse spaces
    scuola = re.sub(rf"[{re.escape(string.punctuation)}]", " ", scuola)
    scuola = re.sub(r'\s+', ' ', scuola).strip()

    # Normalize ELEMENTARI to EL
    scuola = re.sub(r'\b(ELEMENTARI|ELEM|ELE|EL)\b', 'EL', scuola)

    # Remove trailing single uppercase letter ONLY at the end
    scuola = re.sub(r'\s+([A-Z])$', '', scuola)

    # Extract "EL <variable length name>" and ignore known suffixes
    match = re.match(r'^(EL(?:\s+\w+)*?)(?:\s+(?:REF|PICCOLO|PRIMAVERA|SU|GIU).*)?$', scuola)
    if match:
        scuola = match.group(1)

    # Remove known locality words if at the end
    localities = ['SESTO', 'CAMPI', 'CAVALLINA', 'BARBERINO', 'CALENZANO', 'SIGNA', 'MULINO']
    scuola_words = scuola.strip().split()
    if scuola_words and scuola_words[-1] in localities:
        scuola = ' '.join(scuola_words[:-1])

    # Process ragionesociale
    ragione_sociale = str(row["ragionesociale"]).strip()
    ragione_sociale = unicodedata.normalize('NFD', ragione_sociale)
    ragione_sociale = ''.join(c for c in ragione_sociale if unicodedata.category(c) != 'Mn')
    ragione_sociale = ragione_sociale.upper()
    ragione_sociale = re.sub(rf"[{re.escape(string.punctuation)}]", " ", ragione_sociale)
    ragione_sociale = re.sub(r'\s+', ' ', ragione_sociale)
    ragione_sociale = re.sub(r'\bCOMUNE DI\b', '', ragione_sociale).strip()

    # Final cleaned result
    scuola_cleaned = f"{scuola.strip()} a {ragione_sociale}"

    return scuola_cleaned

def add_macrocategoria_column(df: pd.DataFrame, macrocategorie_path: str) -> pd.DataFrame:
    """
    Adds a 'macrocategoria' column to the given DataFrame by matching 'piatto' values
    with those in the macrocategorie Excel file.

    Parameters:
        df (pd.DataFrame): The input DataFrame (from file A), must contain a 'piatto' column.
        macrocategorie_path (str): Path to Excel file (file B) with 'Piatto' and 'Macrocategoria' columns.

    Returns:
        pd.DataFrame: The original DataFrame with an added 'macrocategoria' column.
    """

    # Read the macrocategorie mapping file
    macro_df = pd.read_excel(macrocategorie_path)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()
    macro_df.columns = macro_df.columns.str.strip().str.lower()

    # Ensure both columns are lowercase and stripped for consistent matching
    df["piatto"] = df["piatto"].astype(str).str.strip().str.upper()
    macro_df["piatto"] = macro_df["piatto"].astype(str).str.strip().str.upper()

    # Merge on 'piatto' to bring in 'macrocategoria'
    merged_df = df.merge(
        macro_df[["piatto", "macrocategoria"]],
        how="left",
        on="piatto"
    )

    return merged_df

def excel2influxdb(path, macrocategorie_path):
    # InfluxDB details
    url = "http://localhost:8086"  # Your InfluxDB URL
    token = "9SUJ_bmJB7eSQz5OWS0nPLClLn2TByE-bnh6hyIjTBbC33mZBvZi51LEPWELdgJpoCXPxKWXs0Bx_CvXQOrSiw=="  # Your InfluxDB API token
    org = "smart_food"  # Your InfluxDB organization
    bucket_name = "smart_food_bucket_2023-2024-2025"  # Your InfluxDB bucket

    client = InfluxDBClient(url=url, token=token, org=org)

    buckets_api = client.buckets_api()
    existing_buckets = buckets_api.find_buckets().buckets
    bucket = next((b for b in existing_buckets if b.name == bucket_name), None)

    if bucket:
        try:
            print(f"Deleting existing bucket '{bucket_name}'...")
            buckets_api.delete_bucket(bucket)
            print(f"✅ Bucket '{bucket_name}' deleted.")
        except Exception as e:
            print(f"Error deleting bucket: {e}")
    else:
        print(f"Bucket '{bucket_name}' not found, creating new one...")

    retention_rule = BucketRetentionRules(type="expire", every_seconds=0)
    try:
        buckets_api.create_bucket(bucket_name=bucket_name, org=org, retention_rules=[retention_rule])
        print(f"✅ Bucket '{bucket_name}' created with no expiration.")
    except Exception as e:
        print(f"Error creating bucket: {e}")

    write_api = client.write_api()

    # Read and clean main file
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
    df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y', dayfirst=True)

    # Drop duplicated rows
    df = df.drop_duplicates()

    # Add macrocategoria
    df = add_macrocategoria_column(df, macrocategorie_path)

    # Group by date, school, and food item to aggregate across classes
    grouped_df = df.groupby(['data', 'scuola', 'ragionesociale', 'gruppopiatto', 'piatto', 'macrocategoria']).agg({
        'presenze': 'sum',
        'porzspreco': 'sum'
    }).reset_index()

    for index, row in grouped_df.iterrows():
        timestamp = row["data"]
        scuola_cleaned = clean_school_name(row)

        point = (
            Point("school_food_waste")  # Changed measurement name to be more descriptive
            .tag("ragionesociale", str(row["ragionesociale"]))
            .tag("scuola", scuola_cleaned)
            .tag("gruppopiatto", str(row["gruppopiatto"]))
            .tag("piatto", str(row["piatto"]))
            .tag("macrocategoria", str(row.get("macrocategoria")))
            .field("presenze", float(row["presenze"]))
            .field("porzspreco", float(row["porzspreco"]))
            .time(timestamp, WritePrecision.S)
        )

        try:
            write_api.write(bucket=bucket_name, org=org, record=point)
        except Exception as e:
            print(f"Error writing point to InfluxDB at row {index}: {e}")

    write_api.flush()
    write_api.__del__()
    client.close()

def csv2influxdb(path):
    # InfluxDB details
    url = "http://localhost:8086"  # Your InfluxDB URL
    token = "9SUJ_bmJB7eSQz5OWS0nPLClLn2TByE-bnh6hyIjTBbC33mZBvZi51LEPWELdgJpoCXPxKWXs0Bx_CvXQOrSiw=="  # Your InfluxDB API token
    org = "smart_food"  # Your InfluxDB organization
    bucket_name = "smart_food_bucket_test"  # Your InfluxDB bucket


    # Load CSV
    df = pd.read_csv(path, sep=';', header=None, names=["timestamp", "percspreco"])

    # Remove time part (set hour to 00:00:00 just to standardize)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.normalize()

    # Create an InfluxDB client
    client = InfluxDBClient(url=url, token=token, org=org)



    #---------------------------------------------------------------------------------------------------------------
    #DELETE THE CONTENTS OF THE BUCKET BEFORE INSERTING THE NEW DATA
    # Get the Buckets API
    buckets_api = client.buckets_api()

    # Check if the bucket exists
    existing_buckets = buckets_api.find_buckets().buckets
    bucket = None
    for b in existing_buckets:
        if b.name == bucket_name:
            bucket = b
            break

    # If the bucket exists, delete it and recreate it with no expiration
    if bucket:
        try:
            print(f"Deleting existing bucket '{bucket_name}'...")
            buckets_api.delete_bucket(bucket)
            print(f"✅ Bucket '{bucket_name}' deleted.")
        except Exception as e:
            print(f"Error deleting bucket: {e}")
    else:
        print(f"Bucket '{bucket_name}' not found, proceeding to create a new one.")

    # Now create the bucket with no expiration (infinite retention)
    retention_rule = BucketRetentionRules(type="expire", every_seconds=0)  # 0 seconds = no expiration

    try:
        # Create the bucket with no expiration
        new_bucket = buckets_api.create_bucket(bucket_name=bucket_name, org=org, retention_rules=[retention_rule])
        print(f"✅ Bucket '{bucket_name}' created with no expiration.")
    except Exception as e:
        print(f"Error creating bucket: {e}")

    #---------------------------------------------------------------------------------------------------------------



    # Get a reference to the write API
    write_api = client.write_api()

    ## Write data
    for index, row in df.iterrows():
        point = (
            Point("your_measurement")
            .field("percspreco", float(row["percspreco"]))
            .time(row["timestamp"], WritePrecision.NS)
        )
        try:
            write_api.write(bucket=bucket_name, org=org, record=point)
        except Exception as e:
            print(f"Error writing point to InfluxDB at row {index}: {e}")

    # Finalize and clean up
    write_api.flush()
    write_api.__del__()  # Optional but ensures cleanup
    client.close()

if __name__ == "__main__":
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Export datas to InfluxDB.')
    parser.add_argument('--path', required=True, type=str, help='path to the file to export.')
    parser.add_argument('--filetype', required=True, type=str, help='type of file to export: csv or excel.')
    parser.add_argument('--macroPath', type=str, default='datas/Piatti_Categorizzati.xlsx', help='path to the file that contains food-macrocategory pairs.')

    args = parser.parse_args()

    if args.filetype == "excel":
        excel2influxdb(args.path, args.macroPath)
    elif args.filetype == "csv":
        csv2influxdb(args.path)
    else:
        print("Error File Type")