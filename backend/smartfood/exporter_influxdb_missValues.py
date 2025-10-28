import pandas as pd
import numpy as np
from influxdb_client import InfluxDBClient, Point, WritePrecision, BucketRetentionRules
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime, timezone
import argparse
import re
import unicodedata
import string
from pypots.imputation import SAITS
from pypots.nn.modules.loss import MSE, MAE
from pypots.optim.adam import Adam
from pypots.utils.random import set_random_seed
import os
import torch
import time

# Set environment variables for better memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ------------------ Helper Functions ------------------

def clean_school_name(row):
    """Clean and standardize school names"""
    scuola = str(row["scuola"]).strip()
    scuola = unicodedata.normalize('NFD', scuola)
    scuola = ''.join(c for c in scuola if unicodedata.category(c) != 'Mn')
    scuola = scuola.upper()
    scuola = re.sub(r'\(.*?\)', '', scuola)
    scuola = scuola.replace("L.L.", "L.").replace(":", ".").replace("SEGNOMIGNO", "SEGROMIGNO")
    scuola = re.sub(r'\d.*$', '', scuola)
    scuola = re.sub(r'\s+(GIU|SU)\b', '', scuola)
    scuola = re.sub(fr"[{re.escape(string.punctuation)}]", " ", scuola)
    scuola = re.sub(r'\s+', ' ', scuola).strip()
    scuola = re.sub(r'\b(ELEMENTARI|ELEM|ELE|EL)\b', 'EL', scuola)
    scuola = re.sub(r'\s+([A-Z])$', '', scuola)
    match = re.match(r'^(EL(?:\s+\w+)*?)(?:\s+(?:REF|PICCOLO|PRIMAVERA|SU|GIU).*)?$', scuola)
    if match:
        scuola = match.group(1)
    localities = ['SESTO', 'CAMPI', 'CAVALLINA', 'BARBERINO', 'CALENZANO', 'SIGNA', 'MULINO']
    parts = scuola.split()
    if parts and parts[-1] in localities:
        scuola = ' '.join(parts[:-1])
    rag = str(row["ragionesociale"]).strip()
    rag = unicodedata.normalize('NFD', rag)
    rag = ''.join(c for c in rag if unicodedata.category(c) != 'Mn')
    rag = rag.upper()
    rag = re.sub(fr"[{re.escape(string.punctuation)}]", " ", rag)
    rag = re.sub(r'\s+', ' ', rag)
    rag = re.sub(r'\bCOMUNE DI\b', '', rag).strip()
    return f"{scuola} a {rag}"

def add_macrocategoria_column(df: pd.DataFrame, macrocategorie_path: str) -> pd.DataFrame:
    """Add macrocategory column based on food item"""
    macro_df = pd.read_excel(macrocategorie_path)
    df.columns = df.columns.str.strip().str.lower()
    macro_df.columns = macro_df.columns.str.strip().str.lower()
    df["piatto"] = df["piatto"].astype(str).str.strip().str.upper()
    macro_df["piatto"] = macro_df["piatto"].astype(str).str.strip().str.upper()
    return df.merge(
        macro_df[["piatto", "macrocategoria"]],
        how="left", on="piatto"
    )

# ------------------ School-Level Processing ------------------

def complete_time_series_school_level(df):
    """Complete time series at school level (without classes)"""
    required_cols = ['date', 'scuola', 'ragionesociale', 'gruppopiatto', 'piatto']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Existing columns: {df.columns.tolist()}")
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    
    all_dates = df['date'].unique()
    all_dates = np.array(sorted(all_dates))
    
    all_schools = df[['scuola', 'ragionesociale']].drop_duplicates()

    unique_rows = df.drop_duplicates(subset=['date', 'gruppopiatto', 'piatto'])
    unique_rows['date'] = pd.to_datetime(unique_rows['date']).dt.normalize()
    date_dishes = {
        date: group[['gruppopiatto', 'piatto']].values.tolist()
        for date, group in unique_rows.groupby('date')
    }

    rows = []
    for date in all_dates:
        date_pd = pd.to_datetime(date).normalize()
        dish_combinations = date_dishes.get(date_pd, [])
        if not dish_combinations:
            print(f"Warning: No dishes found for date {date_pd}")
            continue
            
        for _, school_row in all_schools.iterrows():
            scuola = school_row['scuola']
            ragionesociale = school_row['ragionesociale']
            
            for dish_pair in dish_combinations:
                if len(dish_pair) == 2:
                    rows.append({
                        'date': date_pd,
                        'scuola': scuola,
                        'ragionesociale': ragionesociale,
                        'gruppopiatto': dish_pair[0],
                        'piatto': dish_pair[1]
                    })
    
    if not rows:
        raise ValueError("No rows generated - check if date_dishes contains valid data")
    
    completed_df = pd.DataFrame(rows)
    
    value_cols = ['presenze', 'porzspreco']
    if all(col in df.columns for col in value_cols):
        merged_df = pd.merge(
            completed_df,
            df[required_cols + value_cols],
            on=required_cols,
            how='left'
        )
        return merged_df
    
    return completed_df

def prepare_single_school_data(school_df, all_dates):
    """Prepare data for a single school with all dates"""
    presenze_pivot = school_df.pivot(index='date', columns=['gruppopiatto', 'piatto'], values='presenze')
    porzspreco_pivot = school_df.pivot(index='date', columns=['gruppopiatto', 'piatto'], values='porzspreco')
    
    presenze_pivot = presenze_pivot.reindex(all_dates)
    porzspreco_pivot = porzspreco_pivot.reindex(all_dates)
    
    presenze_np = presenze_pivot.values.T[np.newaxis, :, :]
    porzspreco_np = porzspreco_pivot.values.T[np.newaxis, :, :]
    
    combined = np.concatenate([presenze_np[..., np.newaxis], porzspreco_np[..., np.newaxis]], axis=-1)
    
    n_dishes = combined.shape[1]
    final_data = combined.reshape(1, len(all_dates), n_dishes*2)
    
    return final_data

def run_imputation_for_single_school(school_data, n_steps):
    """Run imputation for a single school's data"""
    imputer = SAITS(
        n_steps=n_steps,
        n_features=school_data.shape[2],
        n_layers=2,
        d_model=16,
        n_heads=2,
        d_k=8,
        d_v=8,
        d_ffn=32,
        dropout=0.1,
        attn_dropout=0.1,
        diagonal_attention_mask=True,
        ORT_weight=1,
        MIT_weight=1,
        batch_size=1,
        epochs=50,
        patience=5,
        training_loss=MAE,
        validation_metric=MSE,
        optimizer=Adam,
        num_workers=0,
        device='cuda',
        verbose=False
    )
    
    dataset = {
        "X": school_data,
        "X_ori": school_data.copy(),
        "missing_mask": ~np.isnan(school_data)
    }
    
    imputer.fit(dataset)
    imputed_data = imputer.impute(dataset)
    
    return imputed_data

def add_imputed_values_to_df(school_df, imputed_school, all_dates):
    """Add imputed values back to the school DataFrame"""
    n_dishes = len(school_df[['gruppopiatto', 'piatto']].drop_duplicates())
    imputed_reshaped = imputed_school.reshape(1, len(all_dates), n_dishes, 2)
    
    presenze_imputed = imputed_reshaped[0, :, :, 0].T
    porzspreco_imputed = imputed_reshaped[0, :, :, 1].T
    
    dishes = school_df[['gruppopiatto', 'piatto']].drop_duplicates().values
    dish_to_idx = {tuple(dish): idx for idx, dish in enumerate(dishes)}
    
    for idx, row in school_df.iterrows():
        dish_key = (row['gruppopiatto'], row['piatto'])
        dish_idx = dish_to_idx[dish_key]
        date_idx = np.where(all_dates == row['date'])[0][0]
        
        if pd.isna(row['presenze']):
            school_df.at[idx, 'presenze'] = presenze_imputed[dish_idx, date_idx]
        if pd.isna(row['porzspreco']):
            school_df.at[idx, 'porzspreco'] = porzspreco_imputed[dish_idx, date_idx]
    
    return school_df

def load_saits_model(model_dir, n_steps, n_features):
    """Load a pretrained SAITS model"""
    imputer = SAITS(
        n_steps=n_steps,
        n_features=n_features,
        n_layers=2,
        d_model=16,
        n_heads=2,
        d_k=8,
        d_v=8,
        d_ffn=32,
        dropout=0.1,
        attn_dropout=0.1,
        diagonal_attention_mask=True,
        ORT_weight=1,
        MIT_weight=1,
        device='cuda'
    )
    
    imputer.load(model_dir)
    return imputer

# ------------------ Main Function ------------------

def excel2influxdb(path, macrocategorie_path):
    # InfluxDB setup
    url = "http://localhost:8086"
    token = "9SUJ_bmJB7eSQz5OWS0nPLClLn2TByE-bnh6hyIjTBbC33mZBvZi51LEPWELdgJpoCXPxKWXs0Bx_CvXQOrSiw=="
    org = "smart_food"
    bucket_name = "smart_food_bucket_imputed_2023-2024-2025"

    client = InfluxDBClient(url=url, token=token, org=org)
    buckets_api = client.buckets_api()
    
    # Delete and recreate bucket
    existing = buckets_api.find_buckets().buckets
    bucket = next((b for b in existing if b.name == bucket_name), None)
    if bucket:
        buckets_api.delete_bucket(bucket)
    buckets_api.create_bucket(bucket_name=bucket_name, org=org,
                            retention_rules=BucketRetentionRules(type="expire", every_seconds=0))
    
    try:
        # Use synchronous write API with batching
        write_api = client.write_api(write_options=SYNCHRONOUS)
        
        # Read and preprocess
        df = pd.read_excel(path)
        df.columns = df.columns.str.strip().str.lower()                        
        df.columns = df.columns.str.replace(" ", "_").str.replace("-", "_")
        df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y', dayfirst=True)
        df['date'] = df['data'].dt.normalize()
        
        # Aggregate data at school level
        agg_df = df.groupby(['date', 'scuola', 'ragionesociale', 'gruppopiatto', 'piatto'], as_index=False).agg({
            'presenze': 'sum',
            'porzspreco': 'sum'
        })

        num_rows_before = len(agg_df)
        
        if 'date' not in agg_df.columns:
            raise ValueError("'date' column is missing after aggregation.")
        
        # Complete time series
        df_completed = complete_time_series_school_level(agg_df)
        num_rows_after = len(df_completed)
        rows_added = num_rows_after - num_rows_before

        # Imputation if needed
        if rows_added > 0:
            all_dates = df_completed['date'].unique()
            n_steps = len(all_dates)
            
            date_dishes = df_completed.groupby('date')[['gruppopiatto', 'piatto']].apply(
                lambda x: x.drop_duplicates().values.tolist()
            ).to_dict()
            
            schools = df_completed[['scuola', 'ragionesociale']].drop_duplicates()
            imputed_results = []
            
            for school_idx, (scuola, ragionesociale) in enumerate(schools.itertuples(index=False)):
                print(f"Processing school {school_idx+1}/{len(schools)}: {scuola} - {ragionesociale}")
                
                school_rows = []
                for date in all_dates:
                    dish_combinations = date_dishes.get(date, [])
                    for gruppopiatto, piatto in dish_combinations:
                        school_rows.append({
                            'date': date,
                            'scuola': scuola,
                            'ragionesociale': ragionesociale,
                            'gruppopiatto': gruppopiatto,
                            'piatto': piatto
                        })
                
                school_df = pd.DataFrame(school_rows)
                
                school_df = school_df.merge(
                    df_completed[['date', 'scuola', 'ragionesociale', 'gruppopiatto', 'piatto', 'presenze', 'porzspreco']],
                    on=['date', 'scuola', 'ragionesociale', 'gruppopiatto', 'piatto'],
                    how='left'
                )
                
                school_data = prepare_single_school_data(school_df, all_dates)
                
                if args.from_pretrained:
                    dataset = {
                        "X": school_data,
                        "missing_mask": ~np.isnan(school_data)
                    }
                    loaded_imputer = load_saits_model("saved_models/saits_model_school_level.pypots", n_steps, school_data.shape[2])
                    imputed_school = loaded_imputer.impute(dataset)
                else:
                    imputed_school = run_imputation_for_single_school(school_data, n_steps)
                
                school_df = add_imputed_values_to_df(school_df, imputed_school, all_dates)
                imputed_results.append(school_df)
                
                del school_data, imputed_school
                torch.cuda.empty_cache()
            
            df_imputed = pd.concat(imputed_results, ignore_index=True)
        else:
            df_imputed = df_completed
        
        # Add macrocategories
        df_final = add_macrocategoria_column(df_imputed, macrocategorie_path)
        
        # Batch write to InfluxDB
        batch_size = 1000
        points = []
        
        for idx, row in df_final.iterrows():
            point = (
                Point("school_food_waste")
                .tag("scuola", clean_school_name(row))
                .tag("piatto", row.get("piatto"))
                .tag("gruppopiatto", row.get("gruppopiatto"))
                .tag("macrocategoria", row.get("macrocategoria"))
                .field("presenze", float(row.get("presenze")))
                .field("porzspreco", float(row.get("porzspreco")))
                .time(row['date'], WritePrecision.S)
            )
            points.append(point)
            
            if len(points) >= batch_size:
                write_api.write(bucket=bucket_name, org=org, record=points)
                points = []
                time.sleep(0.1)  # Small delay between batches
        
        # Write any remaining points
        if points:
            write_api.write(bucket=bucket_name, org=org, record=points)
            
    finally:
        # Ensure cleanup
        write_api.close()
        client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export school-level food waste data to InfluxDB.')
    parser.add_argument('--path', required=True, type=str, help='path to the Excel file to export.')
    parser.add_argument('--filetype', required=True, type=str, help='type of file to export: csv or excel.')
    parser.add_argument('--macroPath', type=str, default='datas/Piatti_Categorizzati.xlsx', help='path to the file that contains food-macrocategory pairs.')
    parser.add_argument('--from_pretrained', action="store_true", help='use a pretrained SAITS model')
    args = parser.parse_args()

    if args.filetype == "excel":
        excel2influxdb(args.path, args.macroPath)
    elif args.filetype == "csv":
        print("CSV processing not supported for school-level analysis")
    else:
        print("Error: Invalid file type")