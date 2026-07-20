import os
import cv2
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# CONFIG
# ==========================================================

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent

RESULTS_DIR = os.path.join(BASE_DIR, "results")

IMAGE_BASE = os.path.join(
    BASE_DIR,
    "imagenes",
    "top_imagenes"
)

AOI_FILE = os.path.join(
    BASE_DIR,
    "image_aois",
    "image_aoi_seed42.csv"
)

QA_FILE = os.path.join(
    BASE_DIR,
    "fixation_summary_QA.xlsx"
)

QA_SHEET = "Usable"

OUTPUT_DIR = os.path.join(BASE_DIR, "image_heatmaps")
RAW_DIR = os.path.join(OUTPUT_DIR, "raw")
SMOOTH_DIR = os.path.join(OUTPUT_DIR, "smoothed")
OVERLAY_DIR = os.path.join(OUTPUT_DIR, "overlay")

SIGMA = 45

# ==========================================================
# CREATE OUTPUT FOLDERS
# ==========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(SMOOTH_DIR, exist_ok=True)
os.makedirs(OVERLAY_DIR, exist_ok=True)

# ==========================================================
# FIXATION KERNEL
# ==========================================================

FIXATION_SIGMA = 18          # pixels
KERNEL_RADIUS = int(FIXATION_SIGMA * 3)

x = np.arange(-KERNEL_RADIUS, KERNEL_RADIUS + 1)
y = np.arange(-KERNEL_RADIUS, KERNEL_RADIUS + 1)

xx, yy = np.meshgrid(x, y)

GAUSSIAN_KERNEL = np.exp(
    -(xx**2 + yy**2) /
    (2 * FIXATION_SIGMA**2)
)

GAUSSIAN_KERNEL /= GAUSSIAN_KERNEL.max()

# ==========================================================
# LOAD IMAGE AOIS
# ==========================================================

aoi = pd.read_csv(AOI_FILE)

aoi_lookup = {}

for _, row in aoi.iterrows():

    aoi_lookup[row.filename] = {

        "surf_left": row.surf_left,
        "surf_right": row.surf_right,
        "surf_top": row.surf_top,
        "surf_bottom": row.surf_bottom,

        "trial": int(row.trial_index),

        "emotion": row.emotion
    }

# ==========================================================
# LOCATE ORIGINAL IMAGE FILES
# ==========================================================

image_lookup = {}

for root, _, files in os.walk(IMAGE_BASE):

    for f in files:

        if f.lower().endswith((".jpg", ".jpeg", ".png")):

            image_lookup[f] = os.path.join(root, f)

# ==========================================================
# LOAD USABLE MATRIX
# ==========================================================

usable = pd.read_excel(
    QA_FILE,
    sheet_name=QA_SHEET,
    usecols="B:AM",   # Only participant columns U1–U38
    nrows=60          # Only the 60 image rows
)

# image order = AOI order (trial_index)

participant_columns = list(usable.columns)

print("Participants:", participant_columns)

# ==========================================================
# PREPARE STORAGE
# ==========================================================

all_fixations = {}

for filename in aoi.filename:

    img = cv2.imread(image_lookup[filename])

    h, w = img.shape[:2]

    all_fixations[filename] = {

        "width": w,
        "height": h,

        "raw": np.zeros((h, w), np.float32),

        "count": 0
    }

# ==========================================================
# READ EACH PARTICIPANT ONLY ONCE
# ==========================================================

print("\nReading fixation files...")

for participant in participant_columns:

    participant_dir = os.path.join(RESULTS_DIR, participant)

    if not os.path.exists(participant_dir):

        print(f"Missing folder {participant}")
        continue

    fixation_csv = None

    for f in os.listdir(participant_dir):

        if "fixation" in f.lower() and f.endswith(".csv"):

            fixation_csv = os.path.join(participant_dir, f)
            break

    if fixation_csv is None:

        print(f"No fixation csv for {participant}")
        continue

    print(participant)

    df = pd.read_csv(fixation_csv)

    # ------------------------------------------------------

    for _, fix in df.iterrows():

        filename = fix.filename

        if filename not in aoi_lookup:
            continue

        trial = aoi_lookup[filename]["trial"]

        # row in excel (trial 1 = row index 0)
        usable_value = usable.loc[trial - 1, participant]

        if usable_value != 1:
            continue

        info = aoi_lookup[filename]

        surf_left = info["surf_left"]
        surf_right = info["surf_right"]

        surf_top = info["surf_top"]
        surf_bottom = info["surf_bottom"]

        sx = fix.norm_pos_x
        sy = fix.norm_pos_y

        # convert to image-normalized

        ix = (sx - surf_left) / (surf_right - surf_left)
        iy = 1.0 - ((sy - surf_top) / (surf_bottom - surf_top))

        if ix < 0 or ix > 1:
            continue

        if iy < 0 or iy > 1:
            continue

        img_info = all_fixations[filename]

        x = int(ix * (img_info["width"] - 1))
        y = int(iy * (img_info["height"] - 1))

        duration = float(fix.duration)

        raw = img_info["raw"]

        h, w = raw.shape

        left = max(0, x - KERNEL_RADIUS)
        right = min(w, x + KERNEL_RADIUS + 1)

        top = max(0, y - KERNEL_RADIUS)
        bottom = min(h, y + KERNEL_RADIUS + 1)

        kernel_left = left - (x - KERNEL_RADIUS)
        kernel_right = kernel_left + (right - left)

        kernel_top = top - (y - KERNEL_RADIUS)
        kernel_bottom = kernel_top + (bottom - top)

        raw[top:bottom, left:right] += (
            GAUSSIAN_KERNEL[
                kernel_top:kernel_bottom,
                kernel_left:kernel_right
            ] * duration
        )

        img_info["count"] += 1

# ==========================================================
# GENERATE HEATMAPS
# ==========================================================

print("\nGenerating heatmaps...")

ordered = aoi.sort_values("trial_index")

for _, row in ordered.iterrows():

    filename = row.filename

    img_path = image_lookup[filename]

    image = cv2.imread(img_path)

    data = all_fixations[filename]

    raw = data["raw"]

    np.save(

        os.path.join(
            RAW_DIR,
            filename.replace(".jpg", ".npy").replace(".png", ".npy")
        ),

        raw
    )

    smooth = raw.copy()

    np.save(

        os.path.join(
            SMOOTH_DIR,
            filename.replace(".jpg", ".npy").replace(".png", ".npy")
        ),

        smooth
    )

    if smooth.max() > 0:

        norm = smooth / smooth.max()

    else:

        norm = smooth

    heat = plt.cm.jet(norm)[:, :, :3]
    heat = (heat * 255).astype(np.uint8)

    heat = cv2.cvtColor(
        heat,
        cv2.COLOR_RGB2BGR
    )

    overlay = cv2.addWeighted(
        image,
        0.55,
        heat,
        0.45,
        0
    )

    idx = int(row.trial_index)

    cv2.imwrite(

        os.path.join(
            OUTPUT_DIR,
            f"heatmap_{idx:02d}.png"
        ),

        heat
    )

    cv2.imwrite(

        os.path.join(
            OVERLAY_DIR,
            f"overlay_{idx:02d}.png"
        ),

        overlay
    )

    print(
        f"{idx:02d}/60  "
        f"{filename}   "
        f"{data['count']} fixations"
    )

print("\nDone!")