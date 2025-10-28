import pandas as pd
import os

# === SETTINGS ===
years = [2018, 2019, 2020, 2021, 2022]
columns_to_remove = ["insegnante"]
columns_to_rename = {
    "descrdestinazione": "Scuola",
    "descrclasse": "Classe",
    "descrgruppo": "Gruppo piatto",
    "descrpiatto": "Piatto",
    "qtaavanzata": "porzspreco",
    "percavanzata": "percspreco"
}

for year in years:
    input_base = f"datas/Estrazione annuale {year}_old"
    output_base = f"datas/Estrazione annuale {year}"
    os.makedirs(output_base, exist_ok=True)

    for filename in os.listdir(input_base):
        file_path = os.path.join(input_base, filename)
        if not os.path.isfile(file_path):
            continue

        name_no_ext, ext = os.path.splitext(filename)
        ext = ext.lower()

        print(f"\n=== Processing file: {file_path} ===")

        try:
            if ext == ".csv":
                # normal read — let pandas raise on bad lines
                df = pd.read_csv(file_path, sep=None, engine='python')
            elif ext == ".xlsx":
                df = pd.read_excel(file_path)
            else:
                print(f"Skipping unsupported file type: {filename}")
                continue

        except pd.errors.ParserError as e:
            print(f"\n!!! ParserError reading file: {file_path}")
            print(f"Error: {e}")
            # Read raw lines and print problematic ones
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                lines = list(f)
            print(f"Total lines in file: {len(lines)}")
            print("\n--- Showing lines around the error ---")
            # Try to extract line number from error message
            import re
            m = re.search(r"line\s+(\d+)", str(e))
            if m:
                line_num = int(m.group(1))
                start = max(line_num - 3, 0)
                end = min(line_num + 2, len(lines))
                for i in range(start, end):
                    print(f"{i+1:5d}: {lines[i].strip()}")
            else:
                # If we can’t parse the line number, print the first 50 lines
                for i, line in enumerate(lines[:50], start=1):
                    print(f"{i:5d}: {line.strip()}")
            continue  # skip saving this file until fixed

        # Process dataframe
        df = df.drop(columns=[c for c in columns_to_remove if c in df.columns])
        df = df.rename(columns={k: v for k, v in columns_to_rename.items() if k in df.columns})

        subfolder = os.path.join(output_base, name_no_ext)
        os.makedirs(subfolder, exist_ok=True)
        out_file = os.path.join(subfolder, "reporttipo-1-dettagliato.xlsx")
        df.to_excel(out_file, index=False)

        print(f"Processed {filename} → {out_file}")
