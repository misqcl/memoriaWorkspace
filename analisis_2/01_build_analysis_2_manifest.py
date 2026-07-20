from pathlib import Path
import pandas as pd
from analysis_2_common import clean_name, is_no_response, normalize_emotion, analysis_dir

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
OUT = analysis_dir(BASE_DIR)

SUMMARY_FILE = BASE_DIR / "Todos" / "Summary.xlsx"
TODOS_DIR = BASE_DIR / "Todos"
QA_FILE = BASE_DIR / "fixation_summary_QA.xlsx"
QA_SHEET = "Usable"
AOI_FILE = BASE_DIR / "image_aois" / "image_aoi_seed42.csv"
N_IMAGES = 60
N_USERS = 38

VALID_CSV = OUT / "valid_response_observations.csv"
EXCLUDED_CSV = OUT / "excluded_no_response_observations.csv"
QA_OUT = OUT / "valid_response_matrix.xlsx"
SUMMARY_OUT = OUT / "exclusion_summary.xlsx"

users = pd.read_excel(SUMMARY_FILE, header=None, usecols="A", skiprows=1, nrows=N_USERS)
users.columns = ["raw_name"]
users["participant"] = [f"U{i}" for i in range(1, len(users) + 1)]
users["clean_name"] = users["raw_name"].apply(clean_name)

rows = []
for _, user in users.iterrows():
    path = TODOS_DIR / f"respuestas_emociones_{user['clean_name']}.csv"
    if not path.exists():
        print(f"Missing response file: {path.name}")
        continue
    df = pd.read_csv(path).rename(columns={
        "Respuesta": "response_value", "Emocion": "response_emotion_raw", "Archivo": "filename"
    })
    required = {"filename", "response_emotion_raw"}
    if not required.issubset(df.columns):
        raise KeyError(f"{path.name} lacks columns {sorted(required - set(df.columns))}")
    df["participant"] = user["participant"]
    df["is_no_response"] = df["response_emotion_raw"].apply(is_no_response)
    df["response_emotion"] = df["response_emotion_raw"].apply(normalize_emotion)
    df["response_value"] = pd.to_numeric(df.get("response_value"), errors="coerce")
    rows.append(df[["participant", "filename", "response_value", "response_emotion_raw",
                    "response_emotion", "is_no_response"]])

responses = pd.concat(rows, ignore_index=True)
if responses.duplicated(["participant", "filename"]).any():
    raise ValueError("Duplicate participant-image response rows found")

excluded = responses[responses["is_no_response"] | responses["response_emotion"].isna()].copy()
valid_responses = responses[~responses.index.isin(excluded.index)].copy()

image_info = pd.read_csv(AOI_FILE).sort_values("trial_index")
valid_responses = valid_responses.merge(
    image_info[["trial_index", "filename"]].drop_duplicates("filename"),
    on="filename", how="inner", validate="many_to_one"
)
excluded = excluded.merge(
    image_info[["trial_index", "filename"]].drop_duplicates("filename"),
    on="filename", how="left", validate="many_to_one"
)

qa = pd.read_excel(QA_FILE, sheet_name=QA_SHEET, usecols="B:AM", nrows=N_IMAGES)
valid_pairs = set(zip(valid_responses["participant"], valid_responses["trial_index"].astype(int)))
combined = qa.copy()
for participant in combined.columns:
    for idx in combined.index:
        trial = idx + 1
        combined.loc[idx, participant] = int(combined.loc[idx, participant] == 1 and (participant, trial) in valid_pairs)

usable_pairs = []
for participant in combined.columns:
    for idx, value in combined[participant].items():
        if value == 1:
            usable_pairs.append((participant, idx + 1))
usable_pairs = set(usable_pairs)
valid = valid_responses[
    valid_responses.apply(lambda r: (r["participant"], int(r["trial_index"])) in usable_pairs, axis=1)
].copy()

valid.to_csv(VALID_CSV, index=False, encoding="utf-8-sig")
excluded.to_csv(EXCLUDED_CSV, index=False, encoding="utf-8-sig")
with pd.ExcelWriter(QA_OUT, engine="openpyxl") as writer:
    combined.to_excel(writer, sheet_name=QA_SHEET, index=False)

summary = pd.DataFrame({"metric": [
    "Raw response rows", "NR/no-response rows excluded", "Valid emotional responses",
    "Valid gaze + emotional-response pairs", "Participants retained", "Images retained",
    "Exclusion percentage over response rows"
], "value": [
    len(responses), len(excluded), len(valid_responses), len(valid), valid["participant"].nunique(),
    valid["filename"].nunique(), 100 * len(excluded) / len(responses) if len(responses) else float("nan")
]})
by_emotion = valid["response_emotion"].value_counts(dropna=False).rename_axis("response_emotion").reset_index(name="n")
by_participant = excluded.groupby("participant", as_index=False).size().rename(columns={"size": "excluded_nr"})
# Count exclusions by the seed-42 trial order.
excluded_by_trial = (
    excluded
    .dropna(subset=["trial_index"])
    .assign(trial=lambda df: df["trial_index"].astype(int))
    .groupby("trial", as_index=False)
    .size()
    .rename(columns={"size": "excluded_nr"})
)
by_image = (
    image_info[["trial_index"]]
    .drop_duplicates()
    .rename(columns={"trial_index": "trial"})
    .assign(trial=lambda df: df["trial"].astype(int))
    .merge(excluded_by_trial, on="trial", how="left")
    .fillna({"excluded_nr": 0})
    .sort_values("trial")
    .reset_index(drop=True)
)
by_image["excluded_nr"] = by_image["excluded_nr"].astype(int)
with pd.ExcelWriter(SUMMARY_OUT, engine="openpyxl") as writer:
    summary.to_excel(writer, sheet_name="Summary", index=False)
    by_emotion.to_excel(writer, sheet_name="Valid by Emotion", index=False)
    by_participant.to_excel(writer, sheet_name="Excluded by Participant", index=False)
    by_image.to_excel(writer, sheet_name="Excluded by Image", index=False)

print(f"Saved: {VALID_CSV}")
print(f"Saved: {EXCLUDED_CSV}")
print(f"Saved: {QA_OUT}")
print(f"Saved: {SUMMARY_OUT}")
