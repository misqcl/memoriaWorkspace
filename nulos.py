from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
TODOS_DIR = BASE_DIR / "Todos"

csv_files = sorted(
    TODOS_DIR.glob("respuestas_emociones_*.csv")
)

total_nr = 0
total_no_responde = 0

results = []

for csv_file in csv_files:
    df = pd.read_csv(csv_file)

    # Convert every cell to normalized uppercase text.
    normalized = df.astype(str).apply(
        lambda col: col.str.strip().str.upper()
    )

    nr_count = (
        normalized.eq("NR")
        .sum()
        .sum()
    )

    no_responde_count = (
        normalized.eq("NO RESPONDE")
        .sum()
        .sum()
    )

    total_nr += nr_count
    total_no_responde += no_responde_count

    results.append({
        "file": csv_file.name,
        "rows": len(df),
        "NR": nr_count,
        "NO RESPONDE": no_responde_count,
    })


results_df = pd.DataFrame(results)

combined_count = total_nr + total_no_responde


print("\nFILES READ")
print("----------")
print(f"CSV files: {len(csv_files)}")


print("\nMISSING RESPONSE COUNTS")
print("-----------------------")
print(f"NR: {total_nr}")
print(f"NO RESPONDE: {total_no_responde}")


print("\nPER-FILE COUNTS")
print("----------------")
print(
    results_df.to_string(index=False)
)


results_df.to_csv(
    BASE_DIR / "no_response_counts_by_file.csv",
    index=False,
    encoding="utf-8-sig",
)