import os
import time
import numpy as np
import pandas as pd
from pathlib import Path

start = time.perf_counter()

# ======================================================
# PATHS
# ======================================================

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent

RECORDINGS_ROOT = BASE_DIR / "recordings"
EXCEL_FILE = BASE_DIR / "Voluntarios_1.xlsm"

AOI_FILES = {
    42: r"..\image_aois\image_aoi_seed42.csv",
    43: r"..\image_aois\image_aoi_seed43.csv",
    44: r"..\image_aois\image_aoi_seed44.csv",
}

OUTPUT_FOLDER = "../results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

seed42_order = pd.read_csv(AOI_FILES[42])["filename"].tolist()

# ======================================================
# CENTER BIAS CORRECTION
# ======================================================

SKIP_PERCENT = 0.033333333      # Skip first 3.33% of each trial

# ======================================================
# PARTICIPANTS THAT USED OTHER SEEDS
# ======================================================

participant_seed = {
    "U1": 43,
    "U2": 43,
    "U9": 44,
    "U12": 43,
    "U14": 43,
    "U15": 43,
    "U17": 43,
    "U18": 43,
    "U32": 43,
}

# ======================================================
# RECORDINGS
# ======================================================

date_order = [
    "2026_06_26",
    "2026_05_25",
    "2026_05_27",
    "2026_05_29",
    "2026_06_09",
    "2026_06_10",
    "2026_06_11",
    "2026_06_12",
    "2026_06_16",
    "2026_06_17",
    "2026_06_18",
    "2026_06_19",
]

recordings = []

for date in date_order:

    date_path = os.path.join(RECORDINGS_ROOT, date)

    folders = sorted(
        f for f in os.listdir(date_path)
        if os.path.isdir(os.path.join(date_path, f))
    )

    for folder in folders:

        rec_path = os.path.join(date_path, folder)

        recordings.append({
            "folder": rec_path,
            "fix": os.path.join(
                rec_path,
                "surfaces",
                "fixations_on_surface_Surface 1.csv"
            ),
            "ts": os.path.join(
                rec_path,
                "surfaces",
                "world_timestamps.npy"
            )
        })

assert len(recordings) == 38

# ======================================================
# PROCESS PARTICIPANTS
# ======================================================

for i, rec in enumerate(recordings):

    participant = f"U{i+1}"

    seed = participant_seed.get(participant, 42)

    print(f"\nProcessing {participant} (Seed {seed})")

    # --------------------------------------------------

    aoi_master = pd.read_csv(AOI_FILES[seed])

    timing = pd.read_excel(
        EXCEL_FILE,
        sheet_name=participant,
        usecols="L:M"
    )

    timing.columns = [
        "start_frame",
        "duration_frames"
    ]

    timing = timing.dropna().reset_index(drop=True)

    if len(timing) != len(aoi_master):
        print("Timing/AOI mismatch.")
        continue

    # --------------------------------------------------
    # Load timestamps
    # --------------------------------------------------

    if not os.path.exists(rec["fix"]):
        print("Missing fixation file.")
        continue

    if not os.path.exists(rec["ts"]):
        print("Missing timestamp file.")
        continue

    world_ts = np.load(rec["ts"])

    timing["start_frame"] = timing["start_frame"].astype(int)
    timing["duration_frames"] = timing["duration_frames"].astype(int)

    # --------------------------------------------------
    # Skip the first ~3.33% of every trial
    # --------------------------------------------------

    frame_skip = np.round(
        timing["duration_frames"] * SKIP_PERCENT
    ).astype(int)

    timing["start_frame"] += frame_skip

    # Keep the original trial end unchanged
    timing["duration_frames"] -= frame_skip

    max_frame = len(world_ts) - 1

    if timing["start_frame"].max() > max_frame:
        print("Frame outside recording.")
        continue

    timing["end_frame"] = (
        timing["start_frame"] +
        timing["duration_frames"] -
        1
    ).clip(upper=max_frame)

    timing["start_timestamp"] = world_ts[
        timing["start_frame"].to_numpy()
    ]

    timing["end_timestamp"] = world_ts[
        timing["end_frame"].to_numpy()
    ]

    # --------------------------------------------------
    # AOI
    # --------------------------------------------------

    aoi = aoi_master.copy()

    aoi["start_timestamp"] = timing["start_timestamp"]
    aoi["end_timestamp"] = timing["end_timestamp"]

    # --------------------------------------------------
    # Load fixations
    # --------------------------------------------------

    fix = pd.read_csv(

        rec["fix"],

        usecols=[
            "on_surf",
            "fixation_id",
            "start_timestamp",
            "duration",
            "norm_pos_x",
            "norm_pos_y",
        ]

    )

    fix = fix[fix["on_surf"]].copy()

    fix.drop(columns="on_surf", inplace=True)

    # duration is milliseconds

    fix["end_timestamp"] = (
        fix["start_timestamp"] +
        fix["duration"] / 1000.0
    )

    print(f"Surface fixations: {len(fix)}")

    # --------------------------------------------------
    # FILTER
    # --------------------------------------------------

    participant_fixations = []

    for idx, img in enumerate(aoi.itertuples(index=False), start=1):

        mask = (

            fix["start_timestamp"].between(
            img.start_timestamp,
            img.end_timestamp
            )&

            fix["norm_pos_x"].between(
                img.surf_left,
                img.surf_right
            ) &

            fix["norm_pos_y"].between(
                img.surf_top,
                img.surf_bottom
            )

        )

        img_fix = fix.loc[mask].copy()

        if img_fix.empty:
            continue

        # Average all samples belonging to the same fixation
        img_fix = (
            img_fix
            .groupby("fixation_id", as_index=False)
            .agg({
                "start_timestamp": "first",
                "end_timestamp": "first",
                "duration": "first",
                "norm_pos_x": "mean",
                "norm_pos_y": "mean",
            })
        )

        img_fix["filename"] = img.filename
        img_fix["trial"] = img.trial_index

        participant_fixations.append(img_fix)

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    if participant_fixations:

        out = pd.concat(
            participant_fixations,
            ignore_index=True
        )

        out["filename"] = pd.Categorical(
            out["filename"],
            categories=seed42_order,
            ordered=True
        )

        out = out.sort_values(
            ["filename", "start_timestamp"]
        ).reset_index(drop=True)

        trial_map = {
            f: i + 1
            for i, f in enumerate(seed42_order)
        }

        out["trial"] = out["filename"].map(trial_map)

    else:

        out = pd.DataFrame(columns=[
            "trial",
            "filename",
            "fixation_id",
            "start_timestamp",
            "end_timestamp",
            "duration",
            "norm_pos_x",
            "norm_pos_y",
        ])

    participant_folder = os.path.join(
        OUTPUT_FOLDER,
        participant
    )

    os.makedirs(
        participant_folder,
        exist_ok=True
    )

    out.to_csv(

        os.path.join(
            participant_folder,
            "valid_fixations.csv"
        ),

        index=False

    )

    print(
        f"Saved {participant} "
        f"({len(out)} valid fixations)"
    )

# ======================================================

elapsed = time.perf_counter() - start

print("\nFinished!")

print(f"Total time: {elapsed:.2f} seconds")