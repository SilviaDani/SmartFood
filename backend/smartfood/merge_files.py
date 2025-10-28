import os
import pandas as pd

def fix_presenze_row(row):
    """
    Fix rows where 'presenze' column (7th column, 1-based) is not a number,
    by merging adjacent spilled cells in the 'Piatto' and shifting cells left.
    This handles cases where 'presenze' spans multiple columns.
    """
    max_attempts = 5  # Prevent infinite loops, adjust if needed
    for _ in range(max_attempts):
        presenze_val = row.iloc[6]  # 7th column, 0-based index 6
        try:
            # Try to convert presenze to float
            float(str(presenze_val).replace(',', '.'))  # Also handle comma decimal
            # If success, done fixing
            break
        except (ValueError, TypeError):
            # If presenze is string and next cell(s) exist, merge current and next cell into 'Piatto'
            if len(row) > 7:
                # Merge current 6th and 7th cols (Piatto and spillover)
                # Strip and combine with comma
                merged_piatto = str(row.iloc[5]).strip() + ", " + str(row.iloc[6]).strip()
                # Update Piatto (6th col, index 5)
                row.iloc[5] = merged_piatto
                
                # Shift all cells from 7th col (index 6) onward left by one
                for i in range(6, len(row)-1):
                    row.iloc[i] = row.iloc[i+1]
                # Set last column to empty after shift
                row.iloc[-1] = ""
            else:
                # No more columns to merge, break loop to avoid infinite
                break
    return row

def clean_dataframe(df):
    """
    Clean the dataframe by:
    - Fixing spilled 'presenze' cells
    - Stripping whitespace in string columns
    - Ensuring only first 10 columns remain (drop extras)
    """
    # Fix rows recursively for presenze
    df = df.apply(fix_presenze_row, axis=1)

    # Keep only first 10 columns if more exist
    if df.shape[1] > 10:
        df = df.iloc[:, :10]

    # Strip whitespace for object/string columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    return df

def collect_reports(base_dir, target_filename):
    merged_dfs = []
    file_entries = []
    print(f"Scanning directory: {base_dir}")
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == target_filename:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, base_dir)
                parts = rel_path.split(os.sep)
                folder = parts[0] if len(parts) > 1 else ""
                subfolder = parts[1] if len(parts) > 2 else ""
                file_entries.append((folder, subfolder, file_path))
    file_entries.sort()

    for folder, subfolder, file_path in file_entries:
        print(f"Processing file: {file_path}")
        try:
            # Read with header=0 always (first row as columns)
            df = pd.read_excel(file_path, engine="openpyxl", header=0)

            # Drop rows that are exact duplicate of header row (case-insensitive)
            header_lower = [str(h).strip().lower() for h in df.columns]
            def is_header_row(row):
                return all(str(cell).strip().lower() == header_lower[i] for i, cell in enumerate(row))
            df = df.loc[~df.apply(is_header_row, axis=1)]

            # Clean and fix dataframe
            df = clean_dataframe(df)

            merged_dfs.append(df)
            print(f"Added data from: {file_path} (rows: {len(df)})")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    print(f"Finished scanning {base_dir}, found {len(merged_dfs)} files.")
    return merged_dfs


if __name__ == "__main__":
    target_filename = "reporttipo-1-dettagliato.xlsx"
    all_dfs = []

    for year in ["2023", "2024", "2025"]:  # Adjust years as needed
        base_dir = f"datas/Estrazione annuale {year}"
        dfs = collect_reports(base_dir, target_filename)
        all_dfs.extend(dfs)

    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)

        # Optional: final clean/trimming if needed again here

        merged_df.to_excel("datas/merged_reporttipo-1-dettagliato.xlsx", index=False)
        print(f"Merged {len(all_dfs)} files into datas/merged_reporttipo-1-dettagliato.xlsx")
    else:
        print("No files found to merge.")
