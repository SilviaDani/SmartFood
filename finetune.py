from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import pandas as pd
import numpy as np
from importer_influxdb import query 
import itertools

def build_full_timeseries_dataset(
    query_func,
    start_time,
    end_time,
    augment_n=5,
    jitter_std=0.02,
    dropout_rate=0.05,
    magnitude_scale_std=0.05,
):
    query_types = ["xScuola", "xPiattoxScuola", "globale", "xPiattoGlobale"]
    all_dfs = []

    for query_type in query_types:
        df = query_func(query_type, start_time, end_time)

        if "datetime" not in df or "value" not in df:
            raise ValueError(f"[{query_type}] Missing required columns: 'datetime' and 'value'")

        df["timestamp"] = pd.to_datetime(df["datetime"])

        scuola_str = df["scuola"].astype(str) if "scuola" in df.columns else pd.Series([""] * len(df))
        piatto_str = df["gruppopiatto"].astype(str) if "gruppopiatto" in df.columns else pd.Series([""] * len(df))

        def make_item_id(row):
            parts = [query_type]
            if row["scuola"] != "":
                parts.append(row["scuola"])
            if row["piatto"] != "":
                parts.append(row["piatto"])
            if query_type in ["globale", "xPiattoGlobale"]:
                parts.append("globale")
            return "_".join(parts)

        temp_df = pd.DataFrame({"scuola": scuola_str, "piatto": piatto_str})
        df["item_id"] = temp_df.apply(make_item_id, axis=1)

        df["target"] = df["value"]
        base_df = df[["item_id", "timestamp", "target"]].copy()
        all_dfs.append(base_df)

        # Data augmentation
        for i in range(augment_n):
            aug_df = base_df.copy()
            aug_df["target"] *= np.random.normal(loc=1.0, scale=jitter_std, size=len(aug_df))
            for item in aug_df["item_id"].unique():
                scale = np.random.normal(loc=1.0, scale=magnitude_scale_std)
                aug_df.loc[aug_df["item_id"] == item, "target"] *= scale

            mask = np.random.rand(len(aug_df)) > dropout_rate
            aug_df = aug_df[mask]

            aug_df["target"] = aug_df["target"].clip(0, 1)
            aug_df["item_id"] = aug_df["item_id"] + f"_aug{i+1}"
            all_dfs.append(aug_df)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.sort_values(["item_id", "timestamp"], inplace=True)
    combined_df["timestamp"] = (
        pd.to_datetime(combined_df["timestamp"], utc=True)
        .dt.tz_convert(None)
        .astype("datetime64[ns]")
    )
    return TimeSeriesDataFrame(combined_df[["item_id", "timestamp", "target"]])

# === Load or extract data ===
extract_data = False  # set True if you want to re-query InfluxDB

if extract_data:
    data = build_full_timeseries_dataset(
        query_func=query,
        start_time="2023-01-01T00:00:00Z",
        end_time="2024-01-01T00:00:00Z",
        augment_n=1000,
        jitter_std=0.03,
        dropout_rate=0.05,
        magnitude_scale_std=0.05,
    )
    print("Data extraction complete")
    df = data.to_data_frame().reset_index()
    df.to_csv("finetuning_data.csv", index=False)
    print("Data saved to finetuning_data.csv")
else:
    df = pd.read_csv("finetuning_data.csv", parse_dates=["timestamp"])
    print("Loaded existing finetuning_data.csv with", len(df), "rows")
    data = TimeSeriesDataFrame(df)

prediction_length = 7  # days to predict ahead

# Filter series shorter than prediction_length + 1
min_len = prediction_length + 1
df_data = data.to_data_frame().reset_index()
valid_item_ids = (
    df_data.groupby("item_id")["timestamp"]
    .count()
    .loc[lambda x: x >= min_len]
    .index
)
filtered_data = data.loc[valid_item_ids]

# Train-test split
train_data, test_data = filtered_data.train_test_split(prediction_length)

# Hyperparameter grids
model_paths = ["bolt_tiny", "bolt_mini", "bolt_small", "bolt_base"]
fine_tune_lrs = [1e-6, 1e-5, 1e-4]
fine_tune_steps = [20000, 40000, 60000]
batch_sizes = [64, 128]

hyperparameters = {"Chronos": []}

for m, lr, steps, batch in itertools.product(model_paths, fine_tune_lrs, fine_tune_steps, batch_sizes):
    name_suffix = f"{m}_lr{lr}_steps{steps}_bs{batch}"
    hyperparameters["Chronos"].append({
        "model_path": m,
        "fine_tune": True,
        "fine_tune_lr": lr,
        "fine_tune_steps": steps,
        "batch_size": batch,
        "early_stopping": True,
        "lr_scheduler": {
            "name": "MultiStepLR",
            "milestones": [16000, 32000],
            "gamma": 0.1,
        },
        "ag_args": {"name_suffix": name_suffix},
    })

# Create predictor
predictor = TimeSeriesPredictor(
    prediction_length=7,  # your prediction length
    freq="B",  # business day frequency
    quantile_levels=[0.1, 0.5, 0.9],
    path="finetuned_models/"
)

# Fit with manual grid search (no hyperparameter tuning kwargs)
predictor.fit(
    train_data=train_data,
    hyperparameters=hyperparameters,
    time_limit=172 * 60 * 60,  # 172 hours max
    enable_ensemble=False,
    hyperparameter_tune_kwargs=None,
)

# Optional: print leaderboard after training
print(predictor.leaderboard())
