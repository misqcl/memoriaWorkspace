from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent

IMAGE_AOI_FILE = BASE_DIR / "image_aois" / "image_aoi_seed42.csv"
IMAGE_ROOT = BASE_DIR / "imagenes" / "top_imagenes"

GENERAL_HEATMAP_DIR = BASE_DIR / "image_heatmaps" / "raw"
DEEPGAZE_DIR = BASE_DIR / "deepgaze_predictions" / "raw"

RESULTS_DIR = BASE_DIR / "results"
QA_FILE = BASE_DIR / "fixation_summary_QA.xlsx"
QA_SHEET = "Usable"

OUTPUT_DIR = BASE_DIR / "heatmap_similarity_extremes_vs_deepgaze"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = OUTPUT_DIR / "selected_heatmap_similarity_extremes.csv"
ALL_SCORES_CSV = OUTPUT_DIR / "all_heatmap_similarity_scores.csv"

PARTICIPANT_HEATMAP_SIGMA = 18
MIN_PARTICIPANT_FIXATIONS = 5


def normalize_prob(arr: np.ndarray) -> np.ndarray:
    """Convert a map into a finite, non-negative probability distribution."""
    arr = np.asarray(arr, dtype=np.float64)
    arr[~np.isfinite(arr)] = 0.0
    arr = np.maximum(arr, 0.0)
    total = arr.sum()
    if total <= 0:
        return np.zeros_like(arr, dtype=np.float64)
    return arr / total


def resize_to_shape(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a 2-D map to the reference shape with bilinear interpolation."""
    if arr.shape == shape:
        return arr.astype(np.float64)
    factors = (shape[0] / arr.shape[0], shape[1] / arr.shape[1])
    return zoom(arr, factors, order=1)


def similarity_metrics(empirical: np.ndarray, deepgaze: np.ndarray) -> tuple[float, float, np.ndarray]:
    """
    Return SIM, CC and the normalized residual map (empirical - DeepGaze).

    SIM = sum(min(P, Q)); range [0, 1], where 1 is identical.
    CC  = Pearson correlation between flattened maps; range [-1, 1].
    """
    empirical_norm = normalize_prob(empirical)
    deepgaze_norm = normalize_prob(deepgaze)

    sim = float(np.minimum(empirical_norm, deepgaze_norm).sum())

    e = empirical_norm.ravel()
    d = deepgaze_norm.ravel()
    e_std = float(e.std())
    d_std = float(d.std())
    if e_std == 0.0 or d_std == 0.0:
        cc = np.nan
    else:
        cc = float(np.corrcoef(e, d)[0, 1])

    diff = empirical_norm - deepgaze_norm
    return sim, cc, diff


def add_combined_rank_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add percentile ranks for SIM and CC and average them.

    Rank aggregation avoids assigning an arbitrary numeric weight to SIM and CC.
    The resulting score lies in (0, 1], and a larger value means that the case
    ranks highly under both similarity measures.
    """
    df = df.copy()
    df["sim_rank"] = df["sim"].rank(method="average", pct=True, ascending=True)
    df["cc_rank"] = df["cc"].rank(method="average", pct=True, ascending=True)
    df["similarity_score"] = df[["sim_rank", "cc_rank"]].mean(axis=1, skipna=True)
    return df


def find_images() -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    for path in IMAGE_ROOT.rglob("*"):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            lookup[path.name] = path
    return lookup


def find_deepgaze(trial: int, filename: str) -> Path | None:
    stem = Path(str(filename)).stem
    expected = DEEPGAZE_DIR / f"{int(trial):02d}_{stem}_deepgaze_density.npy"
    if expected.exists():
        return expected
    matches = sorted(DEEPGAZE_DIR.glob(f"*_{stem}_deepgaze_density.npy"))
    return matches[0] if matches else None


def find_general_heatmap(trial: int, filename: str) -> Path | None:
    stem = Path(str(filename)).stem
    candidates = [
        GENERAL_HEATMAP_DIR / f"{stem}.npy",
        GENERAL_HEATMAP_DIR / f"heatmap_{int(trial):02d}.npy",
        GENERAL_HEATMAP_DIR / f"{int(trial):02d}_{stem}.npy",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(GENERAL_HEATMAP_DIR.glob(f"*{stem}*.npy"))
    return matches[0] if matches else None


def create_participant_heatmap(
    fix_df: pd.DataFrame,
    image_row: pd.Series,
    shape: tuple[int, int],
) -> tuple[np.ndarray, int]:
    """Reconstruct one participant's heatmap for one image."""
    height, width = shape
    heatmap = np.zeros((height, width), dtype=np.float64)

    left = float(image_row["surf_left"])
    right = float(image_row["surf_right"])
    top = float(image_row["surf_top"])
    bottom = float(image_row["surf_bottom"])

    if right <= left or bottom <= top:
        return heatmap, 0

    count = 0
    for _, fixation in fix_df.iterrows():
        sx = pd.to_numeric(fixation.get("norm_pos_x"), errors="coerce")
        sy = pd.to_numeric(fixation.get("norm_pos_y"), errors="coerce")
        if pd.isna(sx) or pd.isna(sy):
            continue

        ix = (float(sx) - left) / (right - left)
        iy = (float(sy) - top) / (bottom - top)

        if 0 <= ix <= 1 and 0 <= iy <= 1:
            x = min(width - 1, max(0, int(round(ix * (width - 1)))))
            y = min(height - 1, max(0, int(round(iy * (height - 1)))))
            heatmap[y, x] += 1.0
            count += 1

    if count > 0:
        heatmap = gaussian_filter(heatmap, sigma=PARTICIPANT_HEATMAP_SIGMA)

    return heatmap, count


def save_panel(
    original_path: Path,
    empirical: np.ndarray,
    deepgaze: np.ndarray,
    diff: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    """Save original, empirical, prediction and residual in one comparison panel."""
    image = Image.open(original_path).convert("RGB")
    empirical_norm = normalize_prob(empirical)
    deepgaze_norm = normalize_prob(deepgaze)

    vmax = float(np.max(np.abs(diff))) if np.isfinite(diff).any() else 1.0
    if vmax <= 0:
        vmax = 1.0

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(image)
    axes[0].set_title("Imagen original")
    axes[0].axis("off")

    axes[1].imshow(image)
    axes[1].imshow(empirical_norm, cmap="inferno", alpha=0.55)
    axes[1].set_title("Mapa empírico")
    axes[1].axis("off")

    axes[2].imshow(image)
    axes[2].imshow(deepgaze_norm, cmap="inferno", alpha=0.55)
    axes[2].set_title("DeepGaze IIE")
    axes[2].axis("off")

    residual_plot = axes[3].imshow(diff, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[3].set_title("Empírico - DeepGaze")
    axes[3].axis("off")
    fig.colorbar(residual_plot, ax=axes[3], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def scan_generalized_heatmaps(image_info: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    print("Scanning generalized image heatmaps...")

    for _, image_row in image_info.iterrows():
        trial = int(image_row["trial_index"])
        filename = str(image_row["filename"])

        empirical_path = find_general_heatmap(trial, filename)
        deepgaze_path = find_deepgaze(trial, filename)
        if empirical_path is None or deepgaze_path is None:
            continue

        empirical = np.load(empirical_path)
        deepgaze = resize_to_shape(np.load(deepgaze_path), empirical.shape)
        sim, cc, _ = similarity_metrics(empirical, deepgaze)

        rows.append({
            "type": "generalized_image",
            "participant": "",
            "trial_index": trial,
            "filename": filename,
            "original_emotion": image_row.get("original_emotion", ""),
            "n_fixations": np.nan,
            "sim": sim,
            "cc": cc,
            "empirical_path": str(empirical_path),
            "deepgaze_path": str(deepgaze_path),
        })

    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError(
            "No generalized heatmaps were found. Check image_heatmaps/raw and deepgaze_predictions/raw."
        )
    return add_combined_rank_score(result)


def scan_participant_heatmaps(image_info: pd.DataFrame) -> pd.DataFrame:
    if not QA_FILE.exists():
        raise FileNotFoundError(f"Missing QA file: {QA_FILE}")

    rows: list[dict] = []
    print("Scanning participant-specific image heatmaps...")
    usable = pd.read_excel(QA_FILE, sheet_name=QA_SHEET, usecols="B:AM", nrows=len(image_info))

    for participant in usable.columns:
        fixation_path = RESULTS_DIR / str(participant) / "valid_fixations.csv"
        if not fixation_path.exists():
            continue

        all_fixations = pd.read_csv(fixation_path)
        required_columns = {"filename", "norm_pos_x", "norm_pos_y"}
        if not required_columns.issubset(all_fixations.columns):
            warnings.warn(f"Skipping {participant}: missing columns in {fixation_path}")
            continue

        for _, image_row in image_info.iterrows():
            trial = int(image_row["trial_index"])
            filename = str(image_row["filename"])

            try:
                if usable.loc[trial - 1, participant] != 1:
                    continue
            except (KeyError, IndexError, TypeError):
                continue

            deepgaze_path = find_deepgaze(trial, filename)
            general_heatmap_path = find_general_heatmap(trial, filename)
            if deepgaze_path is None or general_heatmap_path is None:
                continue

            reference_shape = np.load(general_heatmap_path).shape
            deepgaze = resize_to_shape(np.load(deepgaze_path), reference_shape)

            image_fixations = all_fixations[all_fixations["filename"].astype(str) == filename]
            if image_fixations.empty:
                continue

            empirical, fixation_count = create_participant_heatmap(
                image_fixations,
                image_row,
                reference_shape,
            )
            if fixation_count < MIN_PARTICIPANT_FIXATIONS:
                continue

            sim, cc, _ = similarity_metrics(empirical, deepgaze)
            rows.append({
                "type": "participant_image",
                "participant": str(participant),
                "trial_index": trial,
                "filename": filename,
                "original_emotion": image_row.get("original_emotion", ""),
                "n_fixations": int(fixation_count),
                "sim": sim,
                "cc": cc,
                "empirical_path": str(fixation_path),
                "deepgaze_path": str(deepgaze_path),
            })

    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError(
            "No participant heatmaps were reconstructed. Check results/U*/valid_fixations.csv and the QA file."
        )
    return add_combined_rank_score(result)


def select_extremes(general_df: pd.DataFrame, participant_df: pd.DataFrame) -> pd.DataFrame:
    general_most = general_df.loc[general_df["similarity_score"].idxmax()]
    general_least = general_df.loc[general_df["similarity_score"].idxmin()]
    participant_most = participant_df.loc[participant_df["similarity_score"].idxmax()]
    participant_least = participant_df.loc[participant_df["similarity_score"].idxmin()]

    selected = pd.DataFrame([
        general_most,
        general_least,
        participant_most,
        participant_least,
    ]).copy()
    selected["selection"] = [
        "generalized_most_similar",
        "generalized_least_similar",
        "participant_most_similar",
        "participant_least_similar",
    ]
    return selected


def save_selected_panels(
    selected: pd.DataFrame,
    image_info: pd.DataFrame,
    image_lookup: dict[str, Path],
) -> pd.DataFrame:
    print("Saving visual panels...")
    selected = selected.copy()

    for index, selection in selected.iterrows():
        trial = int(selection["trial_index"])
        filename = str(selection["filename"])
        original_path = image_lookup.get(filename)
        if original_path is None:
            warnings.warn(f"Could not find original image for {filename}")
            continue

        deepgaze = np.load(selection["deepgaze_path"])

        if selection["type"] == "generalized_image":
            empirical = np.load(selection["empirical_path"])
            deepgaze = resize_to_shape(deepgaze, empirical.shape)
        else:
            image_row = image_info[image_info["filename"].astype(str) == filename].iloc[0]
            reference_path = find_general_heatmap(trial, filename)
            if reference_path is None:
                warnings.warn(f"Could not find reference heatmap for {filename}")
                continue
            reference_shape = np.load(reference_path).shape
            deepgaze = resize_to_shape(deepgaze, reference_shape)

            all_fixations = pd.read_csv(selection["empirical_path"])
            image_fixations = all_fixations[all_fixations["filename"].astype(str) == filename]
            empirical, _ = create_participant_heatmap(image_fixations, image_row, reference_shape)

        sim, cc, diff = similarity_metrics(empirical, deepgaze)
        title = (
            f"{selection['selection']} | trial {trial:02d} | {filename} | "
            f"participant={selection['participant'] or 'ALL'} | "
            f"SIM={sim:.4f} | CC={cc:.4f} | score={selection['similarity_score']:.4f}"
        )

        output_path = OUTPUT_DIR / (
            f"{selection['selection']}_trial_{trial:02d}_{Path(filename).stem}.png"
        )
        save_panel(original_path, empirical, deepgaze, diff, title, output_path)
        selected.loc[index, "panel_path"] = str(output_path)

    return selected


def main() -> None:
    if not IMAGE_AOI_FILE.exists():
        raise FileNotFoundError(f"Missing image AOI file: {IMAGE_AOI_FILE}")

    image_info = pd.read_csv(IMAGE_AOI_FILE).sort_values("trial_index")
    if "emotion" in image_info.columns and "original_emotion" not in image_info.columns:
        image_info = image_info.rename(columns={"emotion": "original_emotion"})

    image_lookup = find_images()
    general_df = scan_generalized_heatmaps(image_info)
    participant_df = scan_participant_heatmaps(image_info)

    selected = select_extremes(general_df, participant_df)
    selected = save_selected_panels(selected, image_info, image_lookup)

    all_scores = pd.concat([general_df, participant_df], ignore_index=True)
    all_scores.to_csv(ALL_SCORES_CSV, index=False, encoding="utf-8-sig")
    selected.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    print("\nDone.")
    print(f"Selected summary: {SUMMARY_CSV}")
    print(f"All similarity scores: {ALL_SCORES_CSV}")
    print(f"Visual panels: {OUTPUT_DIR}")
    print("\nSelected cases:")
    print(selected[[
        "selection", "participant", "trial_index", "filename",
        "sim", "cc", "similarity_score", "panel_path"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
