import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import re
import os
from datetime import datetime, timezone
import torch

from importer_influxdb import query
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from chronos import BaseChronosPipeline

def sanitize_filename(name: str) -> str:
    """Replace invalid filesystem characters with underscores."""
    return re.sub(r"[^A-Za-z0-9_\-]", "_", name)

def quantile_loss(y_true: np.ndarray, q_pred: np.ndarray, q: float) -> float:
    """Compute pinball (quantile) loss at level q."""
    diff = y_true - q_pred
    return np.mean(np.maximum(q * diff, (q - 1) * diff))

def coverage_80(y_true: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    """Fraction of true points that lie within [low, high]."""
    return np.mean((y_true >= low) & (y_true <= high))

def interval_width_80(low: np.ndarray, high: np.ndarray) -> float:
    """Average width of the 80% prediction interval."""
    return np.mean(high - low)

def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error: sum |y - ŷ| / sum |y|."""
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return 0.0
    return np.sum(np.abs(y_true - y_pred)) / denom

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Forecast time-series (mean + quantiles) either via a fine-tuned AutoGluon predictor or via Chronos bolt_mini."
    )
    parser.add_argument(
        "--prediction_length", required=True, type=int,
        help="Number of future time points to forecast."
    )
    parser.add_argument(
        "--query_type", required=True, type=str,
        help=(
            "Which grouping to forecast:\n"
            "  • xPiattoxScuola: per-school × per-dish-type\n"
            "  • xScuola: per-school only\n"
            "  • globale: single global series\n"
            "  • xPiattoGlobale: per-dish-type across all schools\n"
            "  • xMacrocategoriaGlobale: per-dish-macrocategory\n"
            "  • xMacrocategoriaxScuola: per-school × per-dish-macrocategory"
        )
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="If set, save train/test/forecast plots as PNG."
    )
    parser.add_argument(
        "--output_dir", type=str, default="plots",
        help="Directory in which to save PNG plots."
    )
    parser.add_argument(
        "--start_date", type=lambda s: s + "T00:00:00Z",
        default="2023-01-01T00:00:00Z",
        help="Start time for the InfluxDB query (YYYY-MM-DD). Auto-appends T00:00:00Z."
    )
    parser.add_argument(
        "--end_date", type=lambda s: s + "T00:00:00Z",
        help="End time for the InfluxDB query (YYYY-MM-DD). Auto-appends T00:00:00Z. If omitted, uses today UTC."
    )
    parser.add_argument(
        "--from_finetuning", action="store_true",
        help="Load your local fine-tuned TimeSeriesPredictor (from finetuned_models/)."
    )
    args = parser.parse_args()

    if args.plot:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.end_date is None:
        args.end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d") + "T00:00:00Z"

    # ──────────────── LOAD MODEL ──────────────────
    using_predictor = False
    predictor = None
    pipeline = None

    if args.from_finetuning:
        # ─── Load fine-tuned TimeSeriesPredictor ─── 
        #BEST1D Chronosbolt_mini_lr1e-05_steps60000_bs128[bolt_mini]
        #BEST B Chronosbolt_mini_lr1e-06_steps40000_bs64[bolt_mini]
        predictor = TimeSeriesPredictor.load("finetuned_modelsB/")
        lb = predictor.leaderboard()
        print(lb[["model", "score_val"]])
        using_predictor = True
        print("Loaded fine-tuned TimeSeriesPredictor from 'finetuned_modelsB/'.")
    else:
        # ─── Load Chronos bolt_mini pipeline directly ───
        pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-mini",
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )
        print("Loaded Chronos bolt_mini via from_pretrained().")

    # ──────────────── QUERY DATA ──────────────────
    df = query(args.query_type, args.start_date, args.end_date)
    print("Query done, returned", len(df), "rows.")
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Will accumulate per-series metrics dictionaries here:
    results = []

    # ──────────────── Single-series forecast helpers ──────────────────
    def forecast_with_predictor(
        df_series: pd.DataFrame, item_id: str
    ) -> dict:
        """
        Given a small DataFrame df_series with columns ['datetime','value'], sorted ascending:
        1) Split off the last L points as test.
        2) Build a TimeSeriesDataFrame from the train portion.
        3) Call predictor.predict(...) to get whatever columns it returns.
        4) Find 'mean' (or *_mean), '0.1' (or *_0.1), '0.5' (or *_0.5), '0.9' (or *_0.9).
        5) Return arrays + metrics.
        """
        df_series = df_series.sort_values("datetime").reset_index(drop=True)
        n = len(df_series)
        L = args.prediction_length
        if n <= L:
            return None  # not enough data

        df_train = df_series.iloc[: (n - L)].reset_index(drop=True)
        df_test = df_series.iloc[n - L :].reset_index(drop=True)

        # Ensure train timestamps are datetime64[ns] without timezone
        ts_index = df_train["datetime"]
        if ts_index.dt.tz is not None:
            ts_index = ts_index.dt.tz_convert(None)

        # Build TimeSeriesDataFrame for training
        ts_train = TimeSeriesDataFrame(
            pd.DataFrame({
                "item_id": item_id,
                "timestamp": ts_index, 
                "target": df_train["value"].astype(float),
            })
        )

        # Run predictor.predict(...)
        forecast_ts = predictor.predict(ts_train)
        pdf = forecast_ts.reset_index()
        # Example pdf.columns: ['item_id','timestamp','mean','0.1','0.5','0.9']

        # ──────────────── Find point-forecast column ("mean" or "*_mean") ────────────────
        # 1) Look for exactly "mean"
        if "mean" in pdf.columns:
            mean_col = "mean"
        else:
            # 2) Otherwise look for anything ending in "_mean"
            mean_matches = [c for c in pdf.columns if c.endswith("_mean")]
            if len(mean_matches) >= 1:
                mean_col = mean_matches[0]
            else:
                raise KeyError(
                    "No 'mean' or '*_mean' column found. Available columns = "
                    f"{pdf.columns.tolist()}"
                )

        # ──────────────── Find 10% quantile ("0.1" or "*_0.1") ────────────────
        if "0.1" in pdf.columns:
            q10_col = "0.1"
        else:
            q10_matches = [c for c in pdf.columns if c.endswith("_0.1")]
            if len(q10_matches) >= 1:
                q10_col = q10_matches[0]
            else:
                raise KeyError(
                    "No '0.1' or '*_0.1' column found. Available columns = "
                    f"{pdf.columns.tolist()}"
                )

        # ──────────────── Find 50% quantile ("0.5" or "*_0.5") ────────────────
        if "0.5" in pdf.columns:
            q50_col = "0.5"
        else:
            q50_matches = [c for c in pdf.columns if c.endswith("_0.5")]
            if len(q50_matches) >= 1:
                q50_col = q50_matches[0]
            else:
                # If 0.5 is missing, we will treat the 'mean' itself as median
                q50_col = mean_col

        # ──────────────── Find 90% quantile ("0.9" or "*_0.9") ────────────────
        if "0.9" in pdf.columns:
            q90_col = "0.9"
        else:
            q90_matches = [c for c in pdf.columns if c.endswith("_0.9")]
            if len(q90_matches) >= 1:
                q90_col = q90_matches[0]
            else:
                raise KeyError(
                    "No '0.9' or '*_0.9' column found. Available columns = "
                    f"{pdf.columns.tolist()}"
                )

        # ──────────────── Keep only the last L rows (forecast window) ────────────────
        pdf_last = pdf.iloc[-L :].reset_index(drop=True)

        # ──────────────── Extract arrays ────────────────
        y_true = df_test["value"].to_numpy().astype(float)
        y_mean = pdf_last[mean_col].to_numpy().astype(float)
        y_q10  = pdf_last[q10_col].to_numpy().astype(float)
        y_q50  = pdf_last[q50_col].to_numpy().astype(float)
        y_q90  = pdf_last[q90_col].to_numpy().astype(float)

        # ──────────────── Compute metrics ────────────────
        mae = np.mean(np.abs(y_true - y_mean))
        rmse = np.sqrt(np.mean((y_true - y_mean) ** 2))
        wape_val = wape(y_true, y_mean)

        ql_10 = quantile_loss(y_true, y_q10, 0.1)
        ql_50 = quantile_loss(y_true, y_q50, 0.5)
        ql_90 = quantile_loss(y_true, y_q90, 0.9)
        coverage_val = coverage_80(y_true, y_q10, y_q90)
        interval_width_val = interval_width_80(y_q10, y_q90)

        return {
            "item_id": item_id,
            "timestamps_train": df_train["datetime"].to_numpy(),
            "values_train": df_train["value"].to_numpy(),
            "timestamps_test": df_test["datetime"].to_numpy(),
            "values_test": y_true,
            "y_mean": y_mean,
            "y_q10": y_q10,
            "y_q50": y_q50,
            "y_q90": y_q90,
            "mae": mae,
            "rmse": rmse,
            "wape": wape_val,
            "ql_10": ql_10,
            "ql_50": ql_50,
            "ql_90": ql_90,
            "coverage_80": coverage_val,
            "interval_width_80": interval_width_val,
        }



    def forecast_with_chronos_pipeline(
        df_series: pd.DataFrame, item_id: str
    ) -> dict:
        """
        Given a small DataFrame df_series with columns ['datetime','value'], sorted ascending:
         - split off the last `L` points as test
         - pass the train portion (just the "value" array) to pipeline.predict_quantiles(...)
         - return a dict containing arrays (train timestamps/values, test timestamps/values, 
           predicted mean, predicted q10,q50,q90) plus computed metrics.
        """
        df_series = df_series.sort_values("datetime").reset_index(drop=True)
        n = len(df_series)
        L = args.prediction_length
        if n <= L:
            return None

        df_train = df_series.iloc[: (n - L)].reset_index(drop=True)
        df_test = df_series.iloc[n - L :].reset_index(drop=True)

        # Chronos pipeline expects a 1D torch tensor for `context`
        x_train = torch.tensor(df_train["value"].to_numpy().astype(float))
        quantiles, means = pipeline.predict_quantiles(
            context=x_train,
            prediction_length=L,
            quantile_levels=[0.1, 0.5, 0.9],
        )
        # quantiles shape = (1, L, 3), means shape = (1, L)

        forecast_index = df_test["datetime"].to_numpy()
        low    = quantiles[0, :, 0].numpy()
        median = quantiles[0, :, 1].numpy()
        high   = quantiles[0, :, 2].numpy()
        y_mean = means[0].numpy()
        y_true = df_test["value"].to_numpy().astype(float)

        mae = np.mean(np.abs(y_true - y_mean))
        rmse = np.sqrt(np.mean((y_true - y_mean) ** 2))
        wape_val = wape(y_true, y_mean)

        ql_10 = quantile_loss(y_true, low, 0.1)
        ql_50 = quantile_loss(y_true, median, 0.5)
        ql_90 = quantile_loss(y_true, high, 0.9)
        coverage = coverage_80(y_true, low, high)
        interval_width = interval_width_80(low, high)

        return {
            "item_id": item_id,
            "timestamps_train": df_train["datetime"].to_numpy(),
            "values_train": df_train["value"].to_numpy(),
            "timestamps_test": df_test["datetime"].to_numpy(),
            "values_test": y_true,
            "y_mean": y_mean,
            "y_q10": low,
            "y_q50": median,
            "y_q90": high,
            "mae": mae,
            "rmse": rmse,
            "wape": wape_val,
            "ql_10": ql_10,
            "ql_50": ql_50,
            "ql_90": ql_90,
            "coverage_80": coverage,
            "interval_width_80": interval_width,
        }

    # ──────────────── GROUP-BY LOGIC - xPiattoxScuola ──────────────────
    if args.query_type == "xPiattoxScuola":
        school_names = sorted(df["scuola"].dropna().unique().tolist())
        for school in school_names:
            print("Now analyzing school:", school)
            df_school = df[df["scuola"] == school].copy()

            dish_types = sorted(df_school["gruppopiatto"].dropna().unique().tolist())
            for dish_type in dish_types:
                print("  Dish type:", dish_type)
                mask = (df["scuola"] == school) & (df["gruppopiatto"] == dish_type)
                df_dish = df[mask][["datetime", "value"]].copy()
                df_dish.dropna(subset=["value"], inplace=True)
                df_dish["value"] = pd.to_numeric(df_dish["value"], errors="coerce")
                df_dish.dropna(subset=["value"], inplace=True)
                df_dish.reset_index(drop=True, inplace=True)

                if len(df_dish) <= args.prediction_length:
                    print(
                        f"    Skipping {school}–{dish_type}: "
                        f"only {len(df_dish)} rows (≤{args.prediction_length})"
                    )
                    continue

                item_id = sanitize_filename(f"{school}_{dish_type}")

                if using_predictor:
                    info = forecast_with_predictor(df_dish, item_id)
                else:
                    info = forecast_with_chronos_pipeline(df_dish, item_id)

                if info is None:
                    continue

                results.append({
                    "school": school,
                    "dish_type": dish_type,
                    "query_type": args.query_type,
                    "prediction_length": args.prediction_length,
                    "mae": info["mae"],
                    "rmse": info["rmse"],
                    "wape": info["wape"],
                    "ql_10": info["ql_10"],
                    "ql_50": info["ql_50"],
                    "ql_90": info["ql_90"],
                    "coverage_80": info["coverage_80"],
                    "interval_width_80": info["interval_width_80"],
                })

                if args.plot:
                    plt.figure(figsize=(8, 4))
                    plt.plot(
                        info["timestamps_test"],
                        info["values_test"],
                        color="green",
                        linestyle="--",
                        marker='o',
                        label="Actual"
                    )
                    
                    # Plot predicted mean (red)
                    plt.plot(
                        info["timestamps_test"],
                        info["y_mean"],
                        color="red",
                        marker='x',
                        label="Predicted Mean"
                    )
                    
                    # Plot predicted median (tomato)
                    plt.plot(
                        info["timestamps_test"],
                        info["y_q50"],
                        color="tomato",
                        marker='+',
                        label="Predicted Median"
                    )
                    # Fill 80% PI
                    plt.fill_between(
                        info["timestamps_test"],
                        info["y_q10"],
                        info["y_q90"],
                        color="tomato",
                        alpha=0.3,
                        label="80% PI"
                    )
                    plt.xlabel("Date")
                    plt.ylabel("Value")
                    title = f"{school} – {dish_type}"
                    plt.title(title)
                    plt.legend()
                    plt.grid(True)
                    plt.xticks(rotation=45)
                    filename = sanitize_filename(title) + ".png"
                    filepath = os.path.join(args.output_dir, filename)
                    plt.tight_layout()
                    plt.savefig(filepath)
                    plt.close()

    # ──────────────── xScuola ──────────────────
    elif args.query_type == "xScuola":
        school_names = sorted(df["scuola"].dropna().unique().tolist())
        for school in school_names:
            print("Now analyzing school:", school)
            df_school = df[df["scuola"] == school][["datetime", "value"]].copy()
            df_school.dropna(subset=["value"], inplace=True)
            df_school["value"] = pd.to_numeric(df_school["value"], errors="coerce")
            df_school.dropna(subset=["value"], inplace=True)
            df_school.reset_index(drop=True, inplace=True)

            if len(df_school) <= args.prediction_length:
                print(
                    f"  Skipping {school}: only {len(df_school)} rows (≤ {args.prediction_length})"
                )
                continue

            item_id = sanitize_filename(school)
            if using_predictor:
                info = forecast_with_predictor(df_school, item_id)
            else:
                info = forecast_with_chronos_pipeline(df_school, item_id)

            if info is None:
                continue

            results.append({
                "school": school,
                "dish_type": None,
                "query_type": args.query_type,
                "prediction_length": args.prediction_length,
                "mae": info["mae"],
                "rmse": info["rmse"],
                "wape": info["wape"],
                "ql_10": info["ql_10"],
                "ql_50": info["ql_50"],
                "ql_90": info["ql_90"],
                "coverage_80": info["coverage_80"],
                "interval_width_80": info["interval_width_80"],
            })

            if args.plot:
                plt.figure(figsize=(8, 4))
                plt.plot(
                        info["timestamps_test"],
                        info["values_test"],
                        color="green",
                        linestyle="--",
                        marker='o',
                        label="Actual"
                    )
                    
                # Plot predicted mean (red)
                plt.plot(
                    info["timestamps_test"],
                    info["y_mean"],
                    color="red",
                    marker='x',
                    label="Predicted Mean"
                )
                
                # Plot predicted median (tomato)
                plt.plot(
                    info["timestamps_test"],
                    info["y_q50"],
                    color="tomato",
                    marker='+',
                    label="Predicted Median"
                )

                plt.fill_between(
                    info["timestamps_test"],
                    info["y_q10"],
                    info["y_q90"],
                    color="tomato",
                    alpha=0.3,
                    label="80% PI"
                )
                plt.xlabel("Date")
                plt.ylabel("Value")
                title = f"{school}"
                plt.title(title)
                plt.legend()
                plt.grid(True)
                plt.xticks(rotation=45)
                filename = sanitize_filename(title) + ".png"
                filepath = os.path.join(args.output_dir, filename)
                plt.tight_layout()
                plt.savefig(filepath)
                plt.close()

    # ──────────────── globale ──────────────────
    elif args.query_type == "globale":
        print("Now analyzing global series...")
        df_global = df[["datetime", "value"]].copy()
        df_global.dropna(subset=["value"], inplace=True)
        df_global["value"] = pd.to_numeric(df_global["value"], errors="coerce")
        df_global.dropna(subset=["value"], inplace=True)
        df_global.reset_index(drop=True, inplace=True)

        if len(df_global) <= args.prediction_length:
            print(
                f"  Skipping global: only {len(df_global)} rows (≤ {args.prediction_length})"
            )
        else:
            item_id = "global"
            if using_predictor:
                info = forecast_with_predictor(df_global, item_id)
            else:
                info = forecast_with_chronos_pipeline(df_global, item_id)

            if info is not None:
                results.append({
                    "school": "global",
                    "dish_type": None,
                    "query_type": args.query_type,
                    "prediction_length": args.prediction_length,
                    "mae": info["mae"],
                    "rmse": info["rmse"],
                    "wape": info["wape"],
                    "ql_10": info["ql_10"],
                    "ql_50": info["ql_50"],
                    "ql_90": info["ql_90"],
                    "coverage_80": info["coverage_80"],
                    "interval_width_80": info["interval_width_80"],
                })

                if args.plot:
                    plt.figure(figsize=(8, 4))
                    plt.plot(
                        info["timestamps_test"],
                        info["values_test"],
                        color="green",
                        linestyle="--",
                        marker='o',
                        label="Actual"
                    )
                    
                    # Plot predicted mean (red)
                    plt.plot(
                        info["timestamps_test"],
                        info["y_mean"],
                        color="red",
                        marker='x',
                        label="Predicted Mean"
                    )
                    
                    # Plot predicted median (tomato)
                    plt.plot(
                        info["timestamps_test"],
                        info["y_q50"],
                        color="tomato",
                        marker='+',
                        label="Predicted Median"
                    )
                    plt.fill_between(
                        info["timestamps_test"],
                        info["y_q10"],
                        info["y_q90"],
                        color="tomato",
                        alpha=0.3,
                        label="80% PI"
                    )
                    plt.xlabel("Date")
                    plt.ylabel("Value")
                    title = "Global"
                    plt.title(title)
                    plt.legend()
                    plt.grid(True)
                    plt.xticks(rotation=45)
                    filename = sanitize_filename(title) + ".png"
                    filepath = os.path.join(args.output_dir, filename)
                    plt.tight_layout()
                    plt.savefig(filepath)
                    plt.close()

    # ──────────────── xPiattoGlobale ──────────────────
    elif args.query_type == "xPiattoGlobale":
        print("Now analyzing dish-type globally (across all schools)…")
        dish_types = sorted(df["gruppopiatto"].dropna().unique().tolist())
        for dish_type in dish_types:
            print("  Dish type:", dish_type)
            df_dish = df[df["gruppopiatto"] == dish_type][["datetime", "value"]].copy()
            df_dish.dropna(subset=["value"], inplace=True)
            df_dish["value"] = pd.to_numeric(df_dish["value"], errors="coerce")
            df_dish.dropna(subset=["value"], inplace=True)
            df_dish.reset_index(drop=True, inplace=True)

            if len(df_dish) <= args.prediction_length:
                print(
                    f"    Skipping {dish_type}: only {len(df_dish)} rows (≤ {args.prediction_length})"
                )
                continue

            item_id = sanitize_filename(f"global_{dish_type}")
            if using_predictor:
                info = forecast_with_predictor(df_dish, item_id)
            else:
                info = forecast_with_chronos_pipeline(df_dish, item_id)

            if info is None:
                continue

            results.append({
                "school": "global",
                "dish_type": dish_type,
                "query_type": args.query_type,
                "prediction_length": args.prediction_length,
                "mae": info["mae"],
                "rmse": info["rmse"],
                "wape": info["wape"],
                "ql_10": info["ql_10"],
                "ql_50": info["ql_50"],
                "ql_90": info["ql_90"],
                "coverage_80": info["coverage_80"],
                "interval_width_80": info["interval_width_80"],
            })

            if args.plot:
                plt.figure(figsize=(8, 4))
                plt.plot(
                    info["timestamps_test"],
                    info["values_test"],
                    color="green",
                    linestyle="--",
                    marker='o',
                    label="Actual"
                )
                
                # Plot predicted mean (red)
                plt.plot(
                    info["timestamps_test"],
                    info["y_mean"],
                    color="red",
                    marker='x',
                    label="Predicted Mean"
                )
                
                # Plot predicted median (tomato)
                plt.plot(
                    info["timestamps_test"],
                    info["y_q50"],
                    color="tomato",
                    marker='+',
                    label="Predicted Median"
                )
                plt.fill_between(
                    info["timestamps_test"],
                    info["y_q10"],
                    info["y_q90"],
                    color="tomato",
                    alpha=0.3,
                    label="80% PI"
                )
                plt.xlabel("Date")
                plt.ylabel("Value")
                title = f"Global – {dish_type}"
                plt.title(title)
                plt.legend()
                plt.grid(True)
                plt.xticks(rotation=45)
                filename = sanitize_filename(title) + ".png"
                filepath = os.path.join(args.output_dir, filename)
                plt.tight_layout()
                plt.savefig(filepath)
                plt.close()
    elif args.query_type == "xMacrocategoriaxScuola":
        school_names = sorted(df["scuola"].dropna().unique().tolist())
        for school in school_names:
            print("Now analyzing school:", school)
            df_school = df[df["scuola"] == school].copy()

            macro_types = sorted(df_school["macrocategoria"].dropna().unique().tolist())
            for macro_type in macro_types:
                print("Macro type:", macro_type)
                mask = (df["scuola"] == school) & (df["macrocategoria"] == macro_type)
                df_macro = df[mask][["datetime", "value"]].copy()
                df_macro.dropna(subset=["value"], inplace=True)
                df_macro["value"] = pd.to_numeric(df_macro["value"], errors="coerce")
                df_macro.dropna(subset=["value"], inplace=True)
                df_macro.reset_index(drop=True, inplace=True)

                if len(df_macro) <= args.prediction_length:
                    print(
                        f"    Skipping {school}–{macro_type}: "
                        f"only {len(df_macro)} rows (≤{args.prediction_length})"
                    )
                    continue

                item_id = sanitize_filename(f"{school}_{macro_type}")

                if using_predictor:
                    info = forecast_with_predictor(df_macro, item_id)
                else:
                    info = forecast_with_chronos_pipeline(df_macro, item_id)

                if info is None:
                    continue

                results.append({
                    "school": school,
                    "macro_type": macro_type,
                    "query_type": args.query_type,
                    "prediction_length": args.prediction_length,
                    "mae": info["mae"],
                    "rmse": info["rmse"],
                    "wape": info["wape"],
                    "ql_10": info["ql_10"],
                    "ql_50": info["ql_50"],
                    "ql_90": info["ql_90"],
                    "coverage_80": info["coverage_80"],
                    "interval_width_80": info["interval_width_80"],
                })

                if args.plot:
                    plt.figure(figsize=(8, 4))
                    plt.plot(
                        info["timestamps_test"],
                        info["values_test"],
                        color="green",
                        linestyle="--",
                        marker='o',
                        label="Actual"
                    )
                    
                    # Plot predicted mean (red)
                    plt.plot(
                        info["timestamps_test"],
                        info["y_mean"],
                        color="red",
                        marker='x',
                        label="Predicted Mean"
                    )
                    
                    # Plot predicted median (tomato)
                    plt.plot(
                        info["timestamps_test"],
                        info["y_q50"],
                        color="tomato",
                        marker='+',
                        label="Predicted Median"
                    )
                    # Fill 80% PI
                    plt.fill_between(
                        info["timestamps_test"],
                        info["y_q10"],
                        info["y_q90"],
                        color="tomato",
                        alpha=0.3,
                        label="80% PI"
                    )
                    plt.xlabel("Date")
                    plt.ylabel("Value")
                    title = f"{school} – {macro_type}"
                    plt.title(title)
                    plt.legend()
                    plt.grid(True)
                    plt.xticks(rotation=45)
                    filename = sanitize_filename(title) + ".png"
                    filepath = os.path.join(args.output_dir, filename)
                    plt.tight_layout()
                    plt.savefig(filepath)
                    plt.close()
    else:
        print("ERROR: query_type not recognized.")

    # ──────────────── SAVE METRICS TO CSV ──────────────────
    df_results = pd.DataFrame(results)
    out_csv = f"forecast_metrics_{args.query_type}_{args.prediction_length}.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"Saved metrics to {out_csv}")
