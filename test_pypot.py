import argparse
import pandas as pd
import numpy as np
from pypots.utils.random import set_random_seed
from pypots.forecasting import TimeMixer
from pypots.imputation import SAITS
from importer_influxdb import query
from datetime import datetime, timezone
import re
import os
import matplotlib.pyplot as plt
import csv
from pypots.nn.modules.loss import MSE, MAE, RMSE
from pypots.optim.adam import Adam

def mae(true, pred):
    mask = ~np.isnan(true)
    return np.nanmean(np.abs(true[mask] - pred[mask]))

def rmse(true, pred):
    mask = ~np.isnan(true)
    return np.sqrt(np.nanmean((true[mask] - pred[mask]) ** 2))

def wape(true, pred):
    mask = ~np.isnan(true)
    if not np.any(mask): return np.nan
    denom = np.sum(np.abs(true[mask]))
    return 0.0 if denom == 0 else np.sum(np.abs(true[mask] - pred[mask])) / denom

def quantile_loss(true, pred, q):
    mask = ~np.isnan(true)
    e = true[mask] - pred[mask]
    return np.mean(np.maximum(q * e, (q - 1) * e))

def coverage_80(true, lower, upper):
    mask = ~np.isnan(true)
    return np.mean((true[mask] >= lower[mask]) & (true[mask] <= upper[mask]))

def interval_width_80(lower, upper):
    mask = ~np.isnan(lower) & ~np.isnan(upper)
    return np.mean(upper[mask] - lower[mask])

def complete_time_series(df, query_type):
    # Normalize datetime column to date
    if 'datetime' in df.columns:
        dt_series = pd.to_datetime(df['datetime'])
        if dt_series.dt.tz is not None:
            dt_series = dt_series.dt.tz_convert(None)
        df['date'] = dt_series.dt.normalize()
    else:
        dt_series = pd.to_datetime(df['date'])
        if dt_series.dt.tz is not None:
            dt_series = dt_series.dt.tz_convert(None)
        df['date'] = dt_series.dt.normalize()

    # Define required fields for each query type
    query_requirements = {
        "xScuola": {
            'group_fields': ['scuola'],
            'fill_fields': ['scuola'],
            'school_based': True
        },
        "xPiattoxScuola": {
            'group_fields': ['scuola', 'gruppopiatto'],
            'fill_fields': ['scuola', 'gruppopiatto'],
            'school_based': True
        },
        "globale": {
            'group_fields': [],
            'fill_fields': [],
            'school_based': False
        },
        "xPiattoGlobale": {
            'group_fields': ['gruppopiatto'],
            'fill_fields': ['gruppopiatto'],
            'school_based': False
        },
        "xMacrocategoriaGlobale": {
            'group_fields': ['macrocategoria'],
            'fill_fields': ['macrocategoria'],
            'school_based': False
        },
        "xMacrocategoriaxScuola": {
            'group_fields': ['scuola', 'macrocategoria'],
            'fill_fields': ['scuola', 'macrocategoria'],
            'school_based': True
        }
    }

    requirements = query_requirements.get(query_type)
    if not requirements:
        raise ValueError(f"Unknown query type: {query_type}")

    group_fields = requirements['group_fields']
    fill_fields = requirements['fill_fields']
    school_based = requirements['school_based']

    # Clean data - remove rows with NaN in required fields
    if group_fields:
        df = df.dropna(subset=group_fields)

    # Get all unique dates
    unique_dates = df['date'].unique()
    
    # Get all unique schools (if school-based query)
    unique_schools = df['scuola'].unique() if school_based else []

    # Handle case where no grouping is needed (globale)
    if not school_based:
        # For non-school queries, just ensure all categories exist for each date
        completed_dfs = []
        for date in unique_dates:
            date_df = df[df['date'] == date]
            
            # Get unique values for each non-school field within this date
            unique_values = {}
            for field in fill_fields:
                unique_values[field] = date_df[field].unique()
            
            # Create all combinations for this date
            if len(fill_fields) == 1:
                combinations = pd.DataFrame({
                    fill_fields[0]: unique_values[fill_fields[0]]
                })
            else:
                combinations = pd.MultiIndex.from_product(
                    [unique_values[field] for field in fill_fields],
                    names=fill_fields
                ).to_frame(index=False)
            
            combinations['date'] = date
            completed_dfs.append(combinations.merge(
                date_df,
                on=['date'] + fill_fields,
                how='left'
            ))
        
        df_completed = pd.concat(completed_dfs, ignore_index=True)
    else:
        # For school-based queries
        completed_dfs = []
        for date in unique_dates:
            date_df = df[df['date'] == date]
            
            # Get unique schools and categories for this date
            date_schools = date_df['scuola'].unique()
            other_fields = [f for f in fill_fields if f != 'scuola']
            
            if not other_fields:  # Just xScuola case
                combinations = pd.DataFrame({
                    'scuola': unique_schools,
                    'date': date
                })
            else:
                # Get unique categories for this date
                date_categories = {}
                for field in other_fields:
                    date_categories[field] = date_df[field].unique()
                
                # Create all school-category combinations for this date
                category_combinations = pd.MultiIndex.from_product(
                    [date_categories[field] for field in other_fields],
                    names=other_fields
                ).to_frame(index=False)
                
                # Cross join with all schools
                school_df = pd.DataFrame({'scuola': unique_schools})
                combinations = school_df.assign(key=1).merge(
                    category_combinations.assign(key=1),
                    on='key'
                ).drop('key', axis=1)
                combinations['date'] = date
            
            # Merge with original data
            completed_dfs.append(combinations.merge(
                date_df,
                on=['date'] + fill_fields,
                how='left'
            ))
        
        df_completed = pd.concat(completed_dfs, ignore_index=True)

    # Clean numerical fields
    numerical_fields = ['porzspreco', 'percspreco', 'presenze', 'value']
    for field in numerical_fields:
        if field in df_completed.columns:
            # Keep only the original values (drop any _x/_y columns from merge)
            orig_col = next((c for c in df_completed.columns 
                           if c.startswith(field) and not c.endswith(('_x','_y'))), field)
            df_completed[field] = df_completed[orig_col]
            # Drop any merge artifacts
            for col in [c for c in df_completed.columns 
                       if c.startswith(field) and c != orig_col]:
                df_completed = df_completed.drop(columns=[col])

    # Standardize value column name
    if 'value' not in df_completed.columns and '_value' in df_completed.columns:
        df_completed.rename(columns={'_value': 'value'}, inplace=True)

    return df_completed

def prepare_data_for_imputation(df_completed, query_type):
    """Convert completed DataFrame to numpy array with NaNs for missing values"""
    # First create the item_id column based on query_type, similar to what you do later
    if query_type == "globale":
        df_completed["item_id"] = "globale"
    elif query_type == "xPiattoxScuola":
        df_completed["item_id"] = df_completed["scuola"].fillna('') + "_" + df_completed["gruppopiatto"].fillna('')
    elif query_type == "xMacrocategoriaxScuola":
        df_completed["item_id"] = df_completed["scuola"].fillna('') + "_" + df_completed["macrocategoria"].fillna('')
    elif query_type == "xScuola":
        df_completed["item_id"] = df_completed["scuola"].fillna('')
    elif query_type == "xPiattoGlobale":
        df_completed["item_id"] = df_completed["gruppopiatto"].fillna('')
    elif query_type == "xMacrocategoriaGlobale":
        df_completed["item_id"] = df_completed["macrocategoria"].fillna('')
    else:
        df_completed["item_id"] = "unknown"
    
    # Ensure we have a target column
    if 'value' in df_completed.columns:
        df_completed.rename(columns={'value': 'target'}, inplace=True)
    elif '_value' in df_completed.columns:
        df_completed.rename(columns={'_value': 'target'}, inplace=True)
    
    # Now pivot
    pivot_df = df_completed.pivot(index='date', columns='item_id', values='target')
    data_np = pivot_df.values
    return np.transpose(data_np, (1, 0))[:, :, np.newaxis]  # Shape: (n_series, n_steps, n_features)

def run_imputation(data_with_nans, n_steps, n_features):
    """Run SAITS imputation with all required parameters."""
    # Reshape data if needed
    if data_with_nans.ndim == 2:
        data_with_nans = data_with_nans[:, :, np.newaxis]

    # Initialize SAITS with all required arguments
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
        batch_size=32,
        epochs=50,
        patience=5,
        training_loss=MAE,
        validation_metric=MSE,
        optimizer=Adam,
        num_workers=0,
        device='cuda',
        verbose=True
    )

    # Create proper input format for SAITS
    dataset = {
        "X": data_with_nans,  # This is the key change - wrap array in dict
        "X_ori": data_with_nans.copy(),  # Optional: original data
        "missing_mask": ~np.isnan(data_with_nans)  # Optional: explicit mask
    }

    # Train and impute
    imputer.fit(dataset)
    imputed_data = imputer.impute(dataset)

    # Verify observed values are unchanged
    mask = ~np.isnan(data_with_nans)
    assert np.allclose(imputed_data[mask], data_with_nans[mask], equal_nan=True), \
           "SAITS altered observed data!"
    
    return imputed_data

def prepare_data_for_forecasting(df_completed, query_type, prediction_length):
    """Prepare data for forecasting, keeping all observations but marking future values for prediction"""
    # Create item_id column based on query_type
    if query_type == "globale":
        df_completed["item_id"] = "globale"
    elif query_type == "xPiattoxScuola":
        df_completed["item_id"] = df_completed["scuola"].fillna('') + "_" + df_completed["gruppopiatto"].fillna('')
    elif query_type == "xMacrocategoriaxScuola":
        df_completed["item_id"] = df_completed["scuola"].fillna('') + "_" + df_completed["macrocategoria"].fillna('')
    elif query_type == "xScuola":
        df_completed["item_id"] = df_completed["scuola"].fillna('')
    elif query_type == "xPiattoGlobale":
        df_completed["item_id"] = df_completed["gruppopiatto"].fillna('')
    elif query_type == "xMacrocategoriaGlobale":
        df_completed["item_id"] = df_completed["macrocategoria"].fillna('')
    else:
        df_completed["item_id"] = "unknown"
    
    # Ensure target column exists
    if 'value' in df_completed.columns:
        df_completed.rename(columns={'value': 'target'}, inplace=True)
    elif '_value' in df_completed.columns:
        df_completed.rename(columns={'_value': 'target'}, inplace=True)
    
    # Pivot the data
    pivot_df = df_completed.pivot(index='date', columns='item_id', values='target')
    
    # Find last non-nan index for each series
    last_valid_indices = pivot_df.apply(lambda col: col.last_valid_index())
    
    # Create mask for values to predict (last prediction_length observations for each series)
    predict_mask = pd.DataFrame(False, index=pivot_df.index, columns=pivot_df.columns)
    for col in pivot_df.columns:
        if pd.notna(last_valid_indices[col]):
            # Mark the last prediction_length observations for prediction
            series_dates = pivot_df.index[pivot_df.index <= last_valid_indices[col]]
            if len(series_dates) > prediction_length:
                predict_start = series_dates[-prediction_length]
                predict_mask.loc[predict_start:, col] = True
    
    # Split into train and predict portions
    train_data = pivot_df.where(~predict_mask)
    predict_data = pivot_df.where(predict_mask)
    
    return train_data, predict_data, last_valid_indices

def run_forecasting(train_data, predict_data, last_valid_indices, prediction_length):
    """Run forecasting for each series up to its last observation"""
    # Prepare data for forecasting model
    train_data_np = train_data.values
    predict_data_np = predict_data.values
    
    # Transpose to (n_series, n_steps, n_features)
    train_data_trans = np.transpose(train_data_np, (1, 0))[:, :, np.newaxis]
    
    n_series, n_steps, n_features = train_data_trans.shape
    
    model = TimeMixer(
        n_steps=n_steps,
        n_features=n_features,
        n_pred_steps=prediction_length,
        n_pred_features=n_features,
        term="short",
        n_layers=5,
        d_model=320,
        d_ffn=640,
        top_k=8,
        dropout=0.15,
        channel_independence=False,
        decomp_method="moving_avg",
        moving_avg=7,
        downsampling_layers=2,
        downsampling_window=2,
        use_norm=True,
        batch_size=32,
        epochs=150,
        patience=20,
        training_loss=RMSE,
        validation_metric=RMSE,
        optimizer=Adam,
        verbose=True,
    )
    
    # Initialize X_pred using the last observed value from train_data
    last_obs = train_data_trans[:, -1:, :]  # shape: (n_series, 1, n_features)
    X_pred = np.repeat(last_obs, prediction_length, axis=1)  # shape: (n_series, pred_step, n_features)

    train_dict = {
        "X": train_data_trans,
        "X_pred": X_pred
    }

    model.fit(train_dict)
    forecast = model.predict({"X": train_data_trans})["forecasting"].squeeze(-1)
    forecast = np.maximum(forecast, 0)  # Ensure non-negative forecasts
    
    # Create DataFrame with forecasts aligned to original dates
    forecast_df = pd.DataFrame(
        np.nan,
        index=predict_data.index,
        columns=predict_data.columns
    )
    
    # Fill in the forecast values for each series
    for i, col in enumerate(predict_data.columns):
        if pd.notna(last_valid_indices[col]):
            series_dates = predict_data.index[predict_data.index <= last_valid_indices[col]]
            if len(series_dates) >= prediction_length:
                forecast_dates = series_dates[-prediction_length:]
                forecast_df.loc[forecast_dates, col] = forecast[i, -prediction_length:]
    
    return forecast_df

def save_plot(school=None, dishtype=None, macrocategory=None, fig=None, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    parts = [sanitize_filename(school) if school else "global"]
    if dishtype: parts.append(sanitize_filename(dishtype))
    if macrocategory: parts.append(sanitize_filename(macrocategory))
    filename = "_".join(parts) + ".png"
    if fig:
        fig.savefig(os.path.join(output_dir, filename))
        plt.close(fig)
    else:
        print(f"No figure to save for {filename}")

def sanitize_filename(name):
    return re.sub(r"[^A-Za-z0-9_\-]", "_", name)

def filter_short_series(df, id_col, time_col, min_length):
    counts = df.groupby(id_col)[time_col].count()
    return df[df[id_col].isin(counts[counts > min_length].index)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_length", required=True, type=int)
    parser.add_argument("--query_type", required=True, type=str)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output_dir", type=str, default="plots")
    parser.add_argument("--start_date", type=str, default="2023-01-01")
    parser.add_argument(
        "--end_date", type=lambda s: s + "T00:00:00Z",
        help="End time for the InfluxDB query (YYYY-MM-DD). Auto-appends T00:00:00Z. If omitted, uses today UTC."
    )
    parser.add_argument("--imputation", action="store_true")
    args = parser.parse_args()

    set_random_seed(17)

    if args.end_date is None:
        args.end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d") + "T00:00:00Z"

    # Load and preprocess data
    df = query(args.query_type, args.start_date, args.end_date)
    
    # Remove rows where macrocategoria is NaN or lowercase
    if 'macrocategoria' in df.columns:
        # Convert to string and strip whitespace
        df['macrocategoria'] = df['macrocategoria'].astype(str).str.strip().str.lower()
    
        # Only remove rows where macrocategoria is exactly 'nan'
        df = df[df['macrocategoria'] != 'nan']

    print("Query done, returned", len(df), "rows.")
    print("Original df data last 10 dates:")
    print(df[df['datetime'].dt.normalize() >= '2025-04-01'])
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
    if args.imputation:
        df_completed = complete_time_series(df, args.query_type)
        num_rows_before = len(df)
        num_rows_after = len(df_completed)
        rows_added = num_rows_after - num_rows_before
        print(f"📊 Imputation check: added {rows_added} rows via completion")

        if rows_added == 0:
            print("⚠️ No rows added — skipping imputation")
            df_completed = df.copy()  # fallback to original
            args.imputation = False
    if args.imputation:
        # Prepare data for imputation
        data_with_nans = prepare_data_for_imputation(df_completed, args.query_type)
        n_series, n_steps, n_features = data_with_nans.shape
        
        # Run imputation
        imputed_data = run_imputation(data_with_nans, n_steps, n_features)
        
        # Convert back to DataFrame format
        imputed_flat = imputed_data.squeeze(-1).T  # Transpose back to (n_steps, n_series)
        # Step 1: Make sure dates are sorted and aligned with imputed data
        dates = sorted(df_completed['date'].unique())
        item_ids = sorted(df_completed['item_id'].unique())

        # Step 2: Build pivot_df_imputed with named index
        pivot_df_imputed = pd.DataFrame(
            imputed_flat,
            index=pd.Index(dates, name='date'),  # <- Explicitly name the index
            columns=item_ids
        )

        # Step 3: Reshape back to long format
        df_completed = pivot_df_imputed.reset_index().melt(id_vars='date', var_name='item_id', value_name='target')
    else:
        # Ensure we keep all columns and create date properly
        if 'datetime' in df.columns:
            df['date'] = pd.to_datetime(df['datetime']).dt.tz_localize(None).dt.normalize()
        df_completed = df.copy()  # Keep all columns
        
        # Ensure we have a target column
        if 'value' in df_completed.columns:
            df_completed.rename(columns={'value': 'target'}, inplace=True)
        elif '_value' in df_completed.columns:
            df_completed.rename(columns={'_value': 'target'}, inplace=True)
        else:
            # Find first numeric column to use as target
            numeric_cols = df_completed.select_dtypes(include=[np.number]).columns
            df_completed.rename(columns={numeric_cols[0]: 'target'}, inplace=True)

        # Set item_id for grouping
        if args.query_type == "globale":
            df_completed["item_id"] = "globale"
        elif args.query_type == "xPiattoxScuola":
            df_completed["item_id"] = df_completed["scuola"].fillna('') + "_" + df_completed["gruppopiatto"].fillna('')
        elif args.query_type == "xMacrocategoriaxScuola":
            df_completed["item_id"] = df_completed["scuola"].fillna('') + "_" + df_completed["macrocategoria"].fillna('')
        elif args.query_type == "xScuola":
            df_completed["item_id"] = df_completed["scuola"].fillna('')
        elif args.query_type == "xPiattoGlobale":
            df_completed["item_id"] = df_completed["gruppopiatto"].fillna('')
        elif args.query_type == "xMacrocategoriaGlobale":
            df_completed["item_id"] = df_completed["macrocategoria"].fillna('')
        else:
            df_completed["item_id"] = "unknown"

    print("Completed df shape:", df_completed.shape)

    min_len = args.prediction_length + 5
    df_filtered = filter_short_series(df_completed, 'item_id', 'date', min_len)
    print("Filtered data shape:", df_filtered.shape)

    pivot_df = df_filtered.pivot(index='date', columns='item_id', values='target').sort_index()
    print("✅ Last date in pivot_df:", pivot_df.index[-1])
    print("Last 10 rows of pivot_df:")
    print(pivot_df.tail(10))

    data_np = pivot_df.values
    train_data_np = data_np[:-args.prediction_length, :]
    test_data_np = data_np[-args.prediction_length:, :]

    train_data = np.transpose(train_data_np, (1, 0))[:, :, np.newaxis]
    test_data = np.transpose(test_data_np, (1, 0))[:, :, np.newaxis]

    n_series, n_steps, n_features = train_data.shape
    pred_step = args.prediction_length

    model = TimeMixer(
        n_steps=n_steps,                   # input sequence length
        n_features=n_features,            # number of features
        n_pred_steps=args.prediction_length,
        n_pred_features=n_features,       # predict all features
        term="short",                     # good for short horizons like 7 steps
        n_layers=5,                       # deeper model for richer features
        d_model=320,                      # wider embedding (you could tune between 64–256)
        d_ffn=640,                        # hidden size in feed-forward net
        top_k=8,                          # top-k mixing channels (tune: 3–10)
        dropout=0.15,                      # regularization
        channel_independence=False,       # allows cross-channel learning
        decomp_method="moving_avg",       # seasonality/trend decomposition
        moving_avg=7,                     # MA window size
        downsampling_layers=2,            # reduces input resolution
        downsampling_window=2,
        use_norm=True,                    # enables normalization
        batch_size=32,
        epochs=150,                       # may vary depending on your dataset
        patience=20,                      # for early stopping
        training_loss=RMSE,        
        validation_metric=RMSE,
        optimizer=Adam,
        verbose=True,
    )

    # Initialize X_pred using the last observed value from train_data
    last_obs = train_data[:, -1:, :]  # shape: (n_series, 1, n_features)
    X_pred = np.repeat(last_obs, pred_step, axis=1)  # shape: (n_series, pred_step, n_features)

    train_dict = {
        "X": train_data,
        "X_pred": X_pred
    }

    model.fit(train_dict)
    forecast = model.predict({"X": train_data})["forecasting"].squeeze(-1)
    forecast = np.maximum(forecast, 0)

    maes = []
    rmses = []
    wapes = []

    # For demo purposes, create dummy quantile forecasts (replace with real quantile forecasts if available)
    forecast_q50 = forecast
    forecast_q10 = forecast * 0.9  # Example lower bound (replace with real)
    forecast_q90 = forecast * 1.1  # Example upper bound (replace with real)

    ql_10_list = []
    ql_50_list = []
    ql_90_list = []
    coverage80_list = []
    interval_width80_list = []

    for i in range(n_series):
        true_vals = test_data[i, :, 0]

        maes.append(mae(true_vals, forecast[i]))
        rmses.append(rmse(true_vals, forecast[i]))
        wapes.append(wape(true_vals, forecast[i]))

        ql_10_list.append(quantile_loss(true_vals, forecast_q10[i], 0.10))
        ql_50_list.append(quantile_loss(true_vals, forecast_q50[i], 0.50))
        ql_90_list.append(quantile_loss(true_vals, forecast_q90[i], 0.90))

        coverage80_list.append(coverage_80(true_vals, forecast_q10[i], forecast_q90[i]))
        interval_width80_list.append(interval_width_80(forecast_q10[i], forecast_q90[i]))

    # Extract school and dish_type from item_id by splitting at first underscore
    schools = []
    dish_types = []
    for item_id in pivot_df.columns:
        parts = item_id.split('_', 1)
        schools.append(parts[0] if len(parts) > 0 else "")
        dish_types.append(parts[1] if len(parts) > 1 else "")

    # Save detailed metrics CSV
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_file = os.path.join(args.output_dir, "forecast_metrics_detailed.csv")
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "school", "dish_type", "query_type", "prediction_length",
            "mae", "rmse", "wape", "ql_10", "ql_50", "ql_90", "coverage_80", "interval_width_80"
        ])
        for i in range(n_series):
            writer.writerow([
                schools[i],
                dish_types[i],
                args.query_type,
                args.prediction_length,
                maes[i],
                rmses[i],
                wapes[i],
                ql_10_list[i],
                ql_50_list[i],
                ql_90_list[i],
                coverage80_list[i],
                interval_width80_list[i]
            ])

    print(f"Saved detailed metrics to {metrics_file}")

    if args.plot:
        # Optional plotting code
        for i, item_id in enumerate(pivot_df.columns):
            fig, ax = plt.subplots()
            ax.plot(pivot_df.index[-args.prediction_length:], test_data[i, :, 0], label="True")
            ax.plot(pivot_df.index[-args.prediction_length:], forecast[i], label="Forecast")
            ax.set_title(f"Forecast for {item_id}")
            ax.legend()
            save_plot(school=schools[i], dishtype=dish_types[i], fig=fig, output_dir=args.output_dir)
