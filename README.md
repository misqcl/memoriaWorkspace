# Repositorio de memoria de Martín Sepúlveda Quintanilla, Atención visual y respuesta emocional a imágenes publicitarias: Eye Tracking y Deep Learning.
# Profesor Guía: Miguel Carrasco Zambrano
# Profesor de Comisión: Jorge Zambrano Ibujes

## Definiciones
Este README de apoyo documentará los pasos necesarios para replicar los códigos utilizados para la obtención de datos, análisis e imágenes del documento de tesis.

> [!IMPORTANT]
> Se recomienda ejecutar los scripts en un entorno virtual. 
---
# Análisis 1 (NR/No responde -> Neutral)
En este primer análisis, las respuestas nulas, consideradas como NR o No Responde, se convirtieron en respuestas neutrales, reduciendo la distribución de los datos, en la segunda parte se incluye un análisis en donde estos datos son ignorados, permitiendo verificar si el aumento de distribución de emociones ayuda a los resultados.

## 1. Estructura esperada del proyecto

La mayoría de los scripts calculan la raíz del proyecto como el directorio padre de la carpeta donde se encuentran. Se recomienda mantener la estructura del repositorio.

### Archivos de entrada que no generan los scripts entregados

Antes de ejecutar el pipeline deben existir:

- `recordings/`: exportaciones de Pupil Core/Pupil Player.
- `Voluntarios_1.xlsm`: información de participantes.
- `image_aois/image_aoi_seed42.csv`, `seed43.csv` y `seed44.csv`.
- `imagenes/top_imagenes/`: imágenes publicitarias.
- `Todos/Summary.xlsx` y archivos `Todos/respuestas_emociones_*.csv`.
- `fixation_summary_QA.xlsx`, hoja `Usable`: matriz de control de calidad participante-imagen, el código `user_summary.py` genera el archivo `fixation_summary.xlsx`, este fué verificado y modificado manualmente para solo mantener los pares usuario-imagen con fijaciones mayores a 20, se puede modificar nuevamente el archivo para reducir o aumentar este número, pero se deberá volver a crear o adaptar la hoja `Usable`, ya que varios códigos dependen de ella.
---

## 2. Instalación de dependencias

```bash
pip install --upgrade pip setuptools wheel
pip install numpy pandas scipy matplotlib pillow imageio tqdm openpyxl opencv-python statsmodels scikit-learn
pip install torch torchvision
pip install mediapipe easyocr deepgaze-pytorch optuna
```

> [!NOTE]
> No se detallará la instalación de EasyOCR y MediaPipe por su complejidad, se recomienda probar independientemente que versiones funcionan correctamente antes de ejecutar los scripts de este repositorio.

### Dependencias de R para GLMM

El script en R fué ejecutado en RStudio, y no se probó su ejecución en Visual Studio Code, se recomienda utilizar esta herramienta para evitar errores.

## 1. Procesar las fijaciones
Utiliza los datos de cada par usuario-imagen registrados manualmente en `Voluntarios_1.xlsm`, considerando frame de inicio de imagen y duración, además del seed correspondiente para cada usuario, y las coordenadas normalizadas de cada imagen y exporta todas las fijaciones válidas, tanto en tiempo como en ubicación.

```bash
cd .\relevant_scripts\
py final.py
```

Entrada principal:

```text
recordings/
Voluntarios_1.xlsm
image_aois/image_aoi_seed42.csv
image_aois/image_aoi_seed43.csv
image_aois/image_aoi_seed44.csv
```

Salida esperada:

```text
results/<participante>/valid_fixations.csv
```

## 2. Generar el resumen de control de calidad

```bash
py user_summary.py
```

Salida:

```text
fixation_summary.xlsx
```

Este archivo sirve como apoyo para revisar las fijaciones. Los análisis posteriores esperan un archivo distinto:

```text
fixation_summary_QA.xlsx
└── hoja: Usable
```

Por tanto, entre este paso y el siguiente existe una **revisión manual**: validar los pares participante-imagen y preparar la matriz `Usable`.

## 3. Generar mapas de calor empíricos


```bash
py create_heatmaps.py
```

Salidas principales:

```text
image_heatmaps/raw/
image_heatmaps/smoothed/
image_heatmaps/overlay/
```

## Paso 4. Calcular métricas globales y estadísticas exploratorias
En este paso se calcula Kruskal-Wallis y las pruebas post-hoc, además de resúmenes de los datos encontrados.

```bash
python "heatmap_analysis_with_statistical_summary.py"
```

Salida principal:

```text
heatmap_analysis_clean.xlsx
```

El archivo contiene, entre otras, las hojas:

```text
Summary
Image Metrics
User Image Metrics
Participant Summary
Original Emotion Summary
Response Emotion Summary
Combined Emotion Summary
Emotion Confusion
Emotion Statistics
Posthoc Tests
```

## 5. Ajustar los modelos generalizados de efectos mixtos
> [!IMPORTANT]
> Se ejecutó el código en RStudio, ejecutar el siguiente script solo si ya ha utilizado R en Visual Studio Code.

```bash
Rscript "glmm_all_visual_metrics_fixed_v2.R"
```

Entrada:

```text
heatmap_analysis_clean.xlsx
└── hoja: User Image Metrics
```

Salida:

```text
glmm_all_visual_metrics/
├── glmm_all_metrics_results.xlsx
├── model_information.csv
├── model_comparisons.csv
├── fixed_effects.csv
├── pairwise_contrasts.csv
├── descriptive_statistics.csv
└── model_summaries.txt
```

Modelos utilizados:

- conteo de fijaciones: binomial negativa;
- tiempo de permanencia: Gamma con enlace log;
- duración media de fijación: Gamma con enlace log;
- densidad de fijaciones: Gamma con enlace log;
- interceptos aleatorios: participante e imagen.

## 6. Detectar AOI semánticas


```bash
py create_semantic_aois_MEDIAPIPE_EASYOCR.py
```

Salidas:

```text
semantic_aois_EOCR/
├── semantic_aois_raw.csv
├── semantic_aois_verified.csv
├── detection_summary.xlsx
└── previews/
```

### Revisión manual obligatoria

El script crea AOI con estado inicial `needs_review`. Antes de calcular métricas:

1. revisar las imágenes de `semantic_aois_EOCR/previews/`;
2. corregir cajas incorrectas si corresponde;
3. actualizar `verification_status` en `semantic_aois_verified.csv`;
4. usar estados aceptados por los scripts posteriores, principalmente `accepted` y `manual`.

## 7. Calcular métricas de AOI y asociación con rostros

```bash
py calculate_aoi_metrics_and_chi_square.py
```

Salidas principales:

```text
aoi_analysis_EOCR/
├── aoi_metrics_long.csv
├── aoi_metrics_wide.csv
├── fixation_aoi_assignments.csv
├── chi_square_cramers_v.csv
├── chi_square_contingency_reported_emotion_x_face_presence.csv
└── aoi_analysis_EOCR.xlsx
```

## 8. Calcular asociación con presencia de texto

```bash
py chi_square_text_presence.py
```

Salidas:

```text
aoi_analysis_EOCR/
├── chi_square_text_presence.csv
├── chi_square_contingency_reported_emotion_x_text_presence.csv
└── chi_square_text_presence.xlsx
```

## 9. Generar predicciones de DeepGaze IIE

```bash
cd ..
cd .\deepgaze\
py generate_deepgaze_predictions.py
```

Salidas:

```text
deepgaze_predictions/
├── raw/
├── log_density/
├── previews/
├── overlay_previews/
├── deepgaze_prediction_summary.csv
└── deepgaze_prediction_summary.xlsx
```

## 10. Evaluar DeepGaze contra los mapas empíricos

```bash
py evaluate_deepgaze_predictions.py
```

Métricas calculadas:

```text
CC
SIM
KL(empírico || predicción)
AUC-Judd
NSS
```

Salidas:

```text
deepgaze_evaluation/
├── deepgaze_metrics_by_image.csv
├── deepgaze_metrics_summary.csv
└── deepgaze_evaluation.xlsx
```

## 11. Seleccionar casos extremos de similitud

```bash
py select_extreme_heatmaps_vs_deepgaze_similarity.py
```

Salidas:

```text
heatmap_similarity_extremes_vs_deepgaze/
├── selected_heatmap_similarity_extremes.csv
├── all_heatmap_similarity_scores.csv
└── imagenes comparativas
```

## 12. Entrenar EfficientNet-B0 con validación cruzada

```bash
py train_efficientnet_b0_cv.py
```

Salidas:

```text
efficientnet_b0_cv/
├── efficientnet_fold_metrics.csv
├── efficientnet_predictions.csv
├── efficientnet_confusion_matrix.csv
├── efficientnet_classification_report.csv
└── efficientnet_b0_cv_results.xlsx
```

## 13. Optimizar EfficientNet-B0 con Optuna
Recordar modificar dentro del código el target a accuracy o f1-macro.
```bash
cd ..
cd .\efficientnet
py optuna_efficientnet_b0_cv.py
```

Salidas:

```text
efficientnet_b0_optuna_accuracy/
├── optuna_study.db
├── optuna_trials.csv
├── best_trial_predictions.csv
├── best_trial_fold_metrics.csv
├── best_trial_confusion_matrix.csv
├── best_trial_classification_report.csv
└── optuna_efficientnet_b0_results.xlsx
```

## 14. Integrar los resultados finales

```bash
py "integrate_final_results.py"
```

Entrada esperada por el script:

```text
heatmap_analysis_clean.xlsx
aoi_analysis_EOCR/aoi_metrics_wide.csv
deepgaze_evaluation/deepgaze_metrics_by_image.csv
efficientnet_b0_optuna/best_trial_predictions.csv
```

Salida:

```text
final_integrated_interpretation/
├── final_integrated_interpretation.xlsx
└── final_integrated_image_level.csv
```

# Análisis 2: omisión de NR/nulos

```bash
cd ..
cd .\analisis_2\ 
```


# 5. Scripts que no forman parte del pipeline principal

Los siguientes scripts son auxiliares, alternativas metodológicas o productos gráficos. No son necesarios para ejecutar el pipeline principal en orden.

`nulos.py`

Auditoría de respuestas `NR` y `NO RESPONDE`. Cuenta y exporta frecuencias, pero **no filtra datos**.

`3x3_heamap.py`

Genera una matriz de confusión visual 3×3 entre emoción original y emoción percibida a partir de `heatmap_analysis_clean.xlsx`.

`effect_comparison_p1.py`

Genera un gráfico comparativo de tamaños de efecto entre emoción original y percibida.

`mixed_effects_all_visual_metrics.py`

Implementación alternativa en Python con modelos lineales mixtos sobre métricas transformadas con `log1p`. No reemplaza de forma equivalente al GLMM de R, que utiliza familias binomial negativa y Gamma apropiadas para cada variable.

---
