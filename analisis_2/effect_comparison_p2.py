import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==============================
# Archivo de resultados
# ==============================
excel_file = "heatmap_analysis/heatmap_analysis_clean_no_nr.xlsx"

# ==============================
# Leer tamaños de efecto
# ==============================
df = pd.read_excel(excel_file, sheet_name="Emotion Statistics")

# Mantener solamente epsilon squared
df = df[df["effect_size_name"] == "epsilon_squared"]

# Traducción de nombres
metric_names = {
    "fixation_density_per_megapixel": "Densidad de fijaciones",
    "mean_fixation_duration_ms": "Duración media de fijación",
    "dwell_time_s": "Tiempo de permanencia",
    "fixation_count": "Número de fijaciones"
}

metric_order = [
    "fixation_density_per_megapixel",
    "mean_fixation_duration_ms",
    "dwell_time_s",
    "fixation_count"
]

emocion_original = []
emocion_percibida = []

for metric in metric_order:

    original = df[
        (df["grouping"] == "original_emotion") &
        (df["metric"] == metric)
    ]["effect_size"].iloc[0]

    perceived = df[
        (df["grouping"] == "response_emotion") &
        (df["metric"] == metric)
    ]["effect_size"].iloc[0]

    emocion_original.append(original)
    emocion_percibida.append(perceived)

metricas = [metric_names[m] for m in metric_order]

# ==============================
# Gráfico
# ==============================

y = np.arange(len(metricas))
altura = 0.35

fig, ax = plt.subplots(figsize=(10,6))

barras_original = ax.barh(
    y - altura/2,
    emocion_original,
    height=altura,
    label="Emoción original"
)

barras_percibida = ax.barh(
    y + altura/2,
    emocion_percibida,
    height=altura,
    label="Emoción percibida"
)

ax.set_yticks(y)
ax.set_yticklabels(metricas)
ax.invert_yaxis()

ax.set_xlabel(r"Tamaño de efecto ($\epsilon^2$)")
ax.set_title(
    "Tamaño de efecto de la emoción sobre el comportamiento visual"
)

ax.legend()
ax.grid(axis="x", linestyle="--", alpha=0.4)

ax.bar_label(
    barras_original,
    fmt="%.4f",
    padding=3,
    fontsize=9
)

ax.bar_label(
    barras_percibida,
    fmt="%.4f",
    padding=3,
    fontsize=9
)

# Ajustar automáticamente el eje X
valor_max = max(max(emocion_original), max(emocion_percibida))
ax.set_xlim(0, valor_max * 1.15)

plt.tight_layout()

plt.savefig(
    "effect_size_comparison_p2.jpg",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nTamaños de efecto (ε²):\n")

for metrica, original, percibida in zip(
    metricas,
    emocion_original,
    emocion_percibida
):
    print(f"{metrica}")
    print(f"  Emoción original : {original:.6f}")
    print(f"  Emoción percibida: {percibida:.6f}\n")