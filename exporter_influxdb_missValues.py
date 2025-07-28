import pandas as pd
import numpy as np
from influxdb_client import InfluxDBClient, Point, WritePrecision, BucketRetentionRules
from datetime import datetime, timezone
import argparse
import re
import unicodedata
import string
from pypots.imputation import SAITS
from pypots.nn.modules.loss import MSE, MAE, RMSE
from pypots.optim.adam import Adam
from pypots.utils.random import set_random_seed
import os

# ------------------ Time Series Completion & Imputation (School Level) ------------------

def complete_time_series_school_level(df):
    """Complete time series at school level (without classes)"""
    required_cols = ['date', 'scuola', 'ragionesociale', 'gruppopiatto', 'piatto']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Convert date and remove timezone
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None).dt.normalize()
    
    # Get all unique dates
    all_dates = df['date'].unique()
    
    # Get all unique dish combinations for each date
    date_dishes = df.groupby('date')[['gruppopiatto', 'piatto']].apply(
        lambda x: x.drop_duplicates().values.tolist()
    ).to_dict()
    
    # Get all unique schools (identified by scuola + ragionesociale)
    schools = df[['scuola', 'ragionesociale']].drop_duplicates()

    completed_rows = []
    
    for date in all_dates:
        # Get the dish combinations for this date
        dish_combinations = date_dishes.get(date, [])
        
        for _, (scuola, ragionesociale) in schools.iterrows():
            for gruppopiatto, piatto in dish_combinations:
                completed_rows.append({
                    'date': date,
                    'scuola': scuola,
                    'ragionesociale': ragionesociale,
                    'gruppopiatto': gruppopiatto,
                    'piatto': piatto
                })
    
    # Create completed DataFrame
    completed_df = pd.DataFrame(completed_rows)
    
    # Merge with original data to preserve existing values
    value_cols = ['presenze', 'porzspreco']  # Only merge these value columns
    if all(col in df.columns for col in value_cols):
        completed_df = completed_df.merge(
            df[required_cols + value_cols],
            on=required_cols,
            how='left'
        )
    
    return completed_df

def prepare_data_for_imputation_school_level(df_completed, target_columns=["presenze", "porzspreco"]):
    """Convert completed DataFrame to numpy array with NaNs for missing values (school level)"""
    # Create unique identifier for each time series (without class)
    df_completed["item_id"] = (
        df_completed["scuola"].fillna('') + "_" +
        df_completed["ragionesociale"].fillna('') + "_" +
        df_completed["gruppopiatto"].fillna('') + "_" +
        df_completed["piatto"].fillna('')
    )
    
    # Check if target columns exist
    missing_cols = [col for col in target_columns if col not in df_completed.columns]
    if missing_cols:
        raise ValueError(f"Target columns {missing_cols} not found. Available columns: {df_completed.columns.tolist()}")
    
    # Prepare data for each target column
    data_arrays = []
    for target in target_columns:
        # Create a working copy and drop duplicates
        df_pivot = df_completed[['date', 'item_id', target]].copy()
        df_pivot = df_pivot.drop_duplicates(subset=['date', 'item_id'], keep='first')
        
        # Pivot to wide format for this target
        pivot_df = df_pivot.pivot(index='date', columns='item_id', values=target)
        data_np = pivot_df.values
        # Transpose to (n_series, n_steps) and add feature dimension
        data_arrays.append(np.transpose(data_np, (1, 0))[:, :, np.newaxis])
    
    # Stack along the feature dimension to get (n_series, n_steps, n_features)
    return np.concatenate(data_arrays, axis=2)

def run_imputation(data_with_nans, n_steps, n_features, save_model_dir="saved_models"):
    """Run SAITS imputation on the prepared data"""
    # Ensure reproducibility
    set_random_seed(42)
    
    # Create directory if it doesn't exist
    os.makedirs(save_model_dir, exist_ok=True)
    
    # Initialize SAITS
    imputer = SAITS(
        n_steps=n_steps,
        n_features=n_features,
        n_layers=2,
        d_model=32,
        n_heads=4,
        d_k=8,
        d_v=8,
        d_ffn=64,
        dropout=0.1,
        attn_dropout=0.1,
        diagonal_attention_mask=True,
        ORT_weight=1,
        MIT_weight=1,
        batch_size=8,
        epochs=150,
        patience=15,
        training_loss=MAE,
        validation_metric=MSE,
        optimizer=Adam,
        num_workers=0,
        device='cuda',
        verbose=True
    )

    # Create proper input format for SAITS
    dataset = {
        "X": data_with_nans,
        "X_ori": data_with_nans.copy(),
        "missing_mask": ~np.isnan(data_with_nans)
    }

    # Train and impute
    imputer.fit(dataset)

    # Save the trained model
    model_save_path = os.path.join(save_model_dir, "saits_model_school_level")
    imputer.save(model_save_path)

    imputed_data = imputer.impute(dataset)

    # Verify observed values are unchanged
    mask = ~np.isnan(data_with_nans)
    assert np.allclose(imputed_data[mask], data_with_nans[mask], equal_nan=True), \
           "SAITS altered observed data!"
    
    return imputed_data

def load_saits_model(model_dir, n_steps, n_features):
    """Load a pretrained SAITS model from a directory."""
    # Initialize a dummy SAITS instance with the expected architecture
    imputer = SAITS(
        n_steps=n_steps,
        n_features=n_features,
        n_layers=2,
        d_model=32,
        n_heads=4,
        d_k=8,
        d_v=8,
        d_ffn=64,
        dropout=0.1,
        attn_dropout=0.1,
        diagonal_attention_mask=True,
        ORT_weight=1,
        MIT_weight=1,
        device='cuda'
    )
    
    # Now load the pretrained weights
    imputer.load(model_dir)
    return imputer

# ------------------ Name Cleaning & Category Mapping ------------------

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

# ------------------ Main: Excel to InfluxDB ------------------

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
                            retention_rules=[BucketRetentionRules(type="expire", every_seconds=0)])
    write_api = client.write_api()

    # Read and preprocess
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip().str.lower()                        
    df.columns = df.columns.str.replace(" ", "_").str.replace("-", "_")
    df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y', dayfirst=True)
    df['date'] = df['data'].dt.normalize()
    
    # First aggregate data at school level (sum across classes)
    agg_df = df.groupby(['date', 'scuola', 'ragionesociale', 'gruppopiatto', 'piatto']).agg({
        'presenze': 'sum',
        'porzspreco': 'sum'
    }).reset_index()

    num_rows_before = len(agg_df)
    
    # Complete time series at school level
    df_completed = complete_time_series_school_level(agg_df)
    num_rows_after = len(df_completed)
    rows_added = num_rows_after - num_rows_before

    # Prepare data for imputation if we added missing rows
    if rows_added > 0:
        # Prepare the numpy array for imputation
        data_np = prepare_data_for_imputation_school_level(df_completed)
        n_series, n_steps, n_features = data_np.shape

        # Run imputation
        if args.from_pretrained:
            dataset = {
                "X": data_np,
                "missing_mask": ~np.isnan(data_np)
            }
            loaded_imputer = load_saits_model("saved_models/saits_model_school_level.pypots", n_steps, n_features)
            imputed_data = loaded_imputer.impute(dataset)
        else:
            imputed_data = run_imputation(data_np, n_steps, n_features)
        
        # Convert imputed data back to DataFrame format
        pivot_df = df_completed.pivot(index='date', columns='item_id', values="presenze")
        dates = pivot_df.index
        item_ids = pivot_df.columns
        
        for i, target in enumerate(["presenze", "porzspreco"]):
            imputed_values = imputed_data[:, :, i]
            imputed_df = pd.DataFrame(imputed_values.T, index=dates, columns=item_ids)
            imputed_long = imputed_df.reset_index().melt(id_vars='date', var_name='item_id', value_name=f'imputed_{target}')
            
            df_completed = df_completed.merge(
                imputed_long,
                on=['date', 'item_id'],
                how='left'
            )
            
            df_completed[target] = df_completed[target].fillna(df_completed[f'imputed_{target}'])
            df_completed.drop(columns=[f'imputed_{target}'], inplace=True)
        
        df_imputed = df_completed
    else:
        df_imputed = df_completed

    # Map macrocategoria
    df_final = add_macrocategoria_column(df_imputed, macrocategorie_path)
    
    # Write to InfluxDB (school level only)
    for idx, row in df_final.iterrows():
        point = (
            Point("school_food_waste")
            .tag("scuola", clean_school_name(row))
            .tag("piatto", row.get("piatto"))
            .tag("gruppopiatto", row.get("gruppopiatto"))
            .tag("macrocategoria", row.get("macrocategoria"))
            .field("presenze", int(row.get("presenze")))
            .field("porzspreco", int(row.get("porzspreco")))
            .time(row['date'], WritePrecision.S)
        )
        write_api.write(bucket=bucket_name, org=org, record=point)

    write_api.flush()
    client.close()

if __name__ == "__main__":
    # Parse command line arguments
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