from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# CONFIG
# ==========================================================

BASE_DIR = Path(r"C:\Users\Martin\Desktop\memoria")

ANALYSIS_DIR = BASE_DIR / "analisis_2"
INPUT_FILE = ANALYSIS_DIR / "heatmap_analysis" / "heatmap_analysis_clean_no_nr.xlsx"

OUTPUT_EMOTION_HEATMAP = (
    ANALYSIS_DIR / "emotion_confusion" / "emotion_original_vs_reported_heatmap_no_nr.png"
)

OUTPUT_EMOTION_HEATMAP.parent.mkdir(parents=True, exist_ok=True)

# ==========================================================
# LOAD DATA
# ==========================================================

df_image = pd.read_excel(
    INPUT_FILE,
    sheet_name="Image Metrics"
)

# ==========================================================
# NORMALIZE EMOTION LABELS
# ==========================================================

def normalize_emotion(value):

    if pd.isna(value):
        return pd.NA

    value = str(value).strip().lower()

    mapping = {
        "negative": "Negative",
        "negativo": "Negative",
        "negativa": "Negative",

        "neutral": "Neutral",
        "neutro": "Neutral",
        "neutra": "Neutral",

        "positive": "Positive",
        "positivo": "Positive",
        "positiva": "Positive"
    }

    return mapping.get(value, value)


df_image["original_normalized"] = (
    df_image["original_emotion"]
    .apply(normalize_emotion)
)

df_image["perceived_normalized"] = (
    df_image["most_common_response_emotion"]
    .apply(normalize_emotion)
)

# Original manual tie resolution is intentionally disabled in analysis_2.
# Ties must be recomputed after NR exclusion.
# # ==========================================================
# # RESOLVE KNOWN TIE TO MATCH THESIS TABLE
# # ==========================================================

# df_image.loc[
#     df_image["filename"] == "1123_orig_1359.jpg",
#     "perceived_normalized"
# ] = "Positive"

# ==========================================================
# CREATE TABLE
# ==========================================================

emotion_order = [
    "Negative",
    "Neutral",
    "Positive"
]

emotion_labels_es = [
    "Negativa",
    "Neutral",
    "Positiva"
]

confusion = pd.crosstab(
    df_image["original_normalized"],
    df_image["perceived_normalized"]
)

confusion = confusion.reindex(
    index=emotion_order,
    columns=emotion_order,
    fill_value=0
)

print("\nConfusion matrix:")
print(confusion)

print("\nColumn totals:")
print(confusion.sum(axis=0))

print("\nTotal images:")
print(confusion.to_numpy().sum())

# ==========================================================
# ROW NORMALIZATION
# ==========================================================

normalized = confusion.div(
    confusion.sum(axis=1),
    axis=0
) * 100

print("\nPercentages:")
print(normalized.round(1))

# ==========================================================
# HEATMAP
# ==========================================================

fig, ax = plt.subplots(
    figsize=(7, 6)
)

im = ax.imshow(
    normalized
)

ax.set_xticks(
    range(len(emotion_order))
)

ax.set_yticks(
    range(len(emotion_order))
)

ax.set_xticklabels(
    emotion_labels_es
)

ax.set_yticklabels(
    emotion_labels_es
)

ax.set_xlabel(
    "Emoción percibida mayoritaria"
)

ax.set_ylabel(
    "Emoción original"
)

ax.set_title(
    "Emoción original y emoción percibida mayoritaria"
)

# ==========================================================
# CELL ANNOTATIONS
# ==========================================================

for i in range(normalized.shape[0]):

    for j in range(normalized.shape[1]):

        percentage = normalized.iloc[i, j]
        count = confusion.iloc[i, j]

        ax.text(
            j,
            i,
            f"{percentage:.1f}%\n(n={count})",
            ha="center",
            va="center"
        )

# ==========================================================
# COLORBAR
# ==========================================================

cbar = fig.colorbar(
    im,
    ax=ax
)

cbar.set_label(
    "Porcentaje de imágenes dentro de cada emoción original"
)

# ==========================================================
# SAVE
# ==========================================================

plt.tight_layout()

plt.savefig(
    OUTPUT_EMOTION_HEATMAP,
    dpi=300,
    bbox_inches="tight"
)

#plt.show()

print(
    f"\nSaved: {OUTPUT_EMOTION_HEATMAP}"
)