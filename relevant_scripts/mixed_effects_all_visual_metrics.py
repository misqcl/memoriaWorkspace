from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


# ==========================================================
# PATHS
# ==========================================================

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent

INPUT_FILE = BASE_DIR / "heatmap_analysis_clean.xlsx"
INPUT_SHEET = "User Image Metrics"

OUTPUT_DIR = BASE_DIR / "mixed_effects_all_visual_metrics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_XLSX = OUTPUT_DIR / "mixed_effects_all_metrics_results.xlsx"
MODEL_INFO_CSV = OUTPUT_DIR / "model_information.csv"
MODEL_COMPARISONS_CSV = OUTPUT_DIR / "model_comparisons.csv"
FIXED_EFFECTS_CSV = OUTPUT_DIR / "fixed_effects.csv"
PAIRWISE_CSV = OUTPUT_DIR / "pairwise_contrasts.csv"
DESCRIPTIVE_CSV = OUTPUT_DIR / "descriptive_statistics.csv"
SUMMARY_TXT = OUTPUT_DIR / "model_summaries.txt"


# ==========================================================
# CONFIGURATION
# ==========================================================

REFERENCE_EMOTION = "Neutral"

EMOTION_ORDER = [
    "Negative",
    "Neutral",
    "Positive",
]

METRICS = {
    "dwell_time_s": {
        "display_name": "Dwell time (s)",
        "analysis_column": "log1p_dwell_time_s",
    },
    "fixation_count": {
        "display_name": "Fixation count",
        "analysis_column": "log1p_fixation_count",
    },
    "mean_fixation_duration_ms": {
        "display_name": "Mean fixation duration (ms)",
        "analysis_column": "log1p_mean_fixation_duration_ms",
    },
    "fixation_density_per_megapixel": {
        "display_name": "Fixation density per megapixel",
        "analysis_column": "log1p_fixation_density_per_megapixel",
    },
}

BASE_REQUIRED_COLUMNS = {
    "participant",
    "filename",
    "original_emotion",
    "response_emotion",
}


# ==========================================================
# DATA PREPARATION
# ==========================================================

def normalize_emotion(value: object) -> object:
    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower()

    mapping = {
        "negative": "Negative",
        "negativo": "Negative",
        "negativa": "Negative",
        "neutral": "Neutral",
        "neutro": "Neutral",
        "neutra": "Neutral",
        "positive": "Positive",
        "positivo": "Positive",
        "positiva": "Positive",
        "no responde": "Neutral",
        "no response": "Neutral",
    }

    return mapping.get(text, str(value).strip())


def resolve_metric_columns(data: pd.DataFrame) -> dict[str, str]:
    """
    Resolve likely aliases used in different versions of the workbook.
    Returns mapping from canonical metric name to actual workbook column.
    """

    aliases = {
        "dwell_time_s": [
            "dwell_time_s",
            "total_dwell_time_s",
            "dwell_time",
        ],
        "fixation_count": [
            "fixation_count",
            "total_fixations",
            "number_of_fixations",
        ],
        "mean_fixation_duration_ms": [
            "mean_fixation_duration_ms",
            "average_fixation_duration_ms",
            "avg_fixation_duration_ms",
        ],
        "fixation_density_per_megapixel": [
            "fixation_density_per_megapixel",
            "mean_fixation_density_per_megapixel",
            "fixation_density",
        ],
    }

    lower_lookup = {
        str(column).strip().lower(): column
        for column in data.columns
    }

    resolved = {}

    for canonical, candidates in aliases.items():
        for candidate in candidates:
            if candidate in data.columns:
                resolved[canonical] = candidate
                break

            lower_candidate = candidate.lower()
            if lower_candidate in lower_lookup:
                resolved[canonical] = lower_lookup[lower_candidate]
                break

        if canonical not in resolved:
            raise KeyError(
                f"Could not find a column for '{canonical}'. "
                f"Tried: {candidates}. "
                f"Available columns: {list(data.columns)}"
            )

    return resolved


def load_data() -> tuple[pd.DataFrame, dict[str, str]]:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    data = pd.read_excel(
        INPUT_FILE,
        sheet_name=INPUT_SHEET,
    )

    missing_base = BASE_REQUIRED_COLUMNS - set(data.columns)

    if missing_base:
        raise KeyError(
            "The input sheet is missing required columns: "
            f"{sorted(missing_base)}\n"
            f"Available columns: {list(data.columns)}"
        )

    metric_columns = resolve_metric_columns(data)

    selected_columns = [
        "participant",
        "filename",
        "original_emotion",
        "response_emotion",
        *metric_columns.values(),
    ]

    data = data[selected_columns].copy()

    rename_map = {
        actual: canonical
        for canonical, actual in metric_columns.items()
        if actual != canonical
    }
    data = data.rename(columns=rename_map)

    data["original_emotion"] = (
        data["original_emotion"].apply(normalize_emotion)
    )

    data["response_emotion"] = (
        data["response_emotion"].apply(normalize_emotion)
    )

    for metric in METRICS:
        data[metric] = pd.to_numeric(
            data[metric],
            errors="coerce",
        )

    data = data.dropna(
        subset=[
            "participant",
            "filename",
            "original_emotion",
            "response_emotion",
        ]
    ).copy()

    data = data[
        data["original_emotion"].isin(EMOTION_ORDER)
        & data["response_emotion"].isin(EMOTION_ORDER)
    ].copy()

    data["participant"] = data["participant"].astype(str)
    data["filename"] = data["filename"].astype(str)

    category_order = [
        REFERENCE_EMOTION,
        "Negative",
        "Positive",
    ]

    data["original_emotion"] = pd.Categorical(
        data["original_emotion"],
        categories=category_order,
    )

    data["response_emotion"] = pd.Categorical(
        data["response_emotion"],
        categories=category_order,
    )

    for metric, config in METRICS.items():
        # All four measures are nonnegative in this project.
        data.loc[data[metric] < 0, metric] = np.nan
        data[config["analysis_column"]] = np.log1p(data[metric])

    if data.empty:
        raise RuntimeError(
            "No valid observations remained after cleaning."
        )

    return data.reset_index(drop=True), metric_columns


# ==========================================================
# MODEL FITTING
# ==========================================================

def fit_mixed_model(
    formula: str,
    data: pd.DataFrame,
    model_name: str,
):
    """
    Fit crossed random-intercept model:
      participant = main grouping factor
      filename = variance component
    """

    model = smf.mixedlm(
        formula=formula,
        data=data,
        groups=data["participant"],
        re_formula="1",
        vc_formula={
            "filename": "0 + C(filename)",
        },
    )

    methods = [
        "lbfgs",
        "powell",
        "cg",
    ]

    last_error = None

    for method in methods:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                result = model.fit(
                    reml=False,
                    method=method,
                    maxiter=3000,
                    disp=False,
                )

            print(
                f"{model_name}: fitted with {method}; "
                f"converged={getattr(result, 'converged', 'unknown')}"
            )

            return result

        except Exception as exc:
            last_error = exc
            print(
                f"{model_name}: optimizer {method} failed: {exc}"
            )

    raise RuntimeError(
        f"All optimizers failed for {model_name}."
    ) from last_error


# ==========================================================
# RESULT HELPERS
# ==========================================================

def likelihood_ratio_test(
    reduced_result,
    full_result,
    metric: str,
    reduced_name: str,
    full_name: str,
) -> dict:
    lr_statistic = 2 * (
        full_result.llf - reduced_result.llf
    )

    df_difference = int(
        full_result.df_modelwc
        - reduced_result.df_modelwc
    )

    if df_difference <= 0:
        p_value = np.nan
    else:
        p_value = chi2.sf(
            max(lr_statistic, 0),
            df_difference,
        )

    return {
        "metric": metric,
        "reduced_model": reduced_name,
        "full_model": full_name,
        "log_likelihood_reduced": reduced_result.llf,
        "log_likelihood_full": full_result.llf,
        "lr_statistic": lr_statistic,
        "df_difference": df_difference,
        "p_value": p_value,
        "significant_0_05": (
            p_value < 0.05
            if pd.notna(p_value)
            else np.nan
        ),
    }


def extract_fixed_effects(
    result,
    metric: str,
    model_name: str,
) -> pd.DataFrame:
    fixed_params = result.fe_params
    fixed_names = list(fixed_params.index)

    covariance = result.cov_params().loc[
        fixed_names,
        fixed_names,
    ]

    rows = []

    for term in fixed_names:
        estimate = float(fixed_params[term])
        standard_error = float(
            np.sqrt(covariance.loc[term, term])
        )

        z_value = (
            estimate / standard_error
            if standard_error > 0
            else np.nan
        )

        p_value = (
            2 * norm.sf(abs(z_value))
            if pd.notna(z_value)
            else np.nan
        )

        ci_lower = estimate - 1.96 * standard_error
        ci_upper = estimate + 1.96 * standard_error

        rows.append(
            {
                "metric": metric,
                "model": model_name,
                "term": term,
                "estimate_log1p_scale": estimate,
                "standard_error": standard_error,
                "z_value": z_value,
                "p_value": p_value,
                "ci_95_lower": ci_lower,
                "ci_95_upper": ci_upper,
                "approx_percent_change": (
                    np.expm1(estimate) * 100
                ),
            }
        )

    return pd.DataFrame(rows)


def coefficient_name(
    predictor: str,
    emotion: str,
) -> str:
    return (
        f"C({predictor}, "
        f"Treatment(reference='{REFERENCE_EMOTION}'))"
        f"[T.{emotion}]"
    )


def linear_contrast(
    result,
    weights: dict[str, float],
) -> tuple[float, float, float, float, float, float]:
    fixed_params = result.fe_params
    fixed_names = list(fixed_params.index)

    covariance = result.cov_params().loc[
        fixed_names,
        fixed_names,
    ]

    contrast = pd.Series(
        0.0,
        index=fixed_names,
    )

    for term, weight in weights.items():
        if term not in contrast.index:
            raise KeyError(
                f"Term not found in model: {term}\n"
                f"Available fixed terms: {fixed_names}"
            )

        contrast.loc[term] = weight

    estimate = float(
        contrast.to_numpy()
        @ fixed_params.to_numpy()
    )

    variance = float(
        contrast.to_numpy()
        @ covariance.to_numpy()
        @ contrast.to_numpy()
    )

    standard_error = np.sqrt(
        max(variance, 0)
    )

    z_value = (
        estimate / standard_error
        if standard_error > 0
        else np.nan
    )

    p_value = (
        2 * norm.sf(abs(z_value))
        if pd.notna(z_value)
        else np.nan
    )

    ci_lower = estimate - 1.96 * standard_error
    ci_upper = estimate + 1.96 * standard_error

    return (
        estimate,
        standard_error,
        z_value,
        p_value,
        ci_lower,
        ci_upper,
    )


def pairwise_emotion_contrasts(
    result,
    metric: str,
    model_name: str,
    predictor: str,
) -> pd.DataFrame:
    negative_term = coefficient_name(
        predictor,
        "Negative",
    )

    positive_term = coefficient_name(
        predictor,
        "Positive",
    )

    comparisons = [
        {
            "comparison": "Negative vs Neutral",
            "weights": {
                negative_term: 1.0,
            },
        },
        {
            "comparison": "Positive vs Neutral",
            "weights": {
                positive_term: 1.0,
            },
        },
        {
            "comparison": "Negative vs Positive",
            "weights": {
                negative_term: 1.0,
                positive_term: -1.0,
            },
        },
    ]

    rows = []

    for item in comparisons:
        (
            estimate,
            standard_error,
            z_value,
            p_value,
            ci_lower,
            ci_upper,
        ) = linear_contrast(
            result,
            item["weights"],
        )

        rows.append(
            {
                "metric": metric,
                "model": model_name,
                "predictor": predictor,
                "comparison": item["comparison"],
                "estimate_log1p_scale": estimate,
                "standard_error": standard_error,
                "z_value": z_value,
                "p_value_raw": p_value,
                "ci_95_lower": ci_lower,
                "ci_95_upper": ci_upper,
                "approx_percent_difference": (
                    np.expm1(estimate) * 100
                ),
            }
        )

    contrasts = pd.DataFrame(rows)

    valid = contrasts["p_value_raw"].notna()

    if valid.any():
        contrasts.loc[
            valid,
            "p_value_bonferroni",
        ] = multipletests(
            contrasts.loc[
                valid,
                "p_value_raw",
            ],
            method="bonferroni",
        )[1]

    contrasts["significant_bonferroni_0_05"] = (
        contrasts["p_value_bonferroni"] < 0.05
    )

    return contrasts


def descriptive_summary(
    data: pd.DataFrame,
    metric: str,
    predictor: str,
) -> pd.DataFrame:
    summary = (
        data
        .groupby(
            predictor,
            observed=True,
        )[metric]
        .agg(
            observations="count",
            mean="mean",
            standard_deviation="std",
            median="median",
            minimum="min",
            maximum="max",
        )
        .reset_index()
    )

    summary.insert(
        0,
        "metric",
        metric,
    )

    summary.insert(
        1,
        "grouping_variable",
        predictor,
    )

    return summary


# ==========================================================
# ANALYSIS FOR ONE METRIC
# ==========================================================

def analyze_metric(
    full_data: pd.DataFrame,
    metric: str,
    analysis_column: str,
    summary_file,
):
    data = full_data.dropna(
        subset=[
            metric,
            analysis_column,
        ]
    ).copy()

    if data.empty:
        raise RuntimeError(
            f"No valid observations available for {metric}."
        )

    original_term = (
        "C(original_emotion, "
        f"Treatment(reference='{REFERENCE_EMOTION}'))"
    )

    response_term = (
        "C(response_emotion, "
        f"Treatment(reference='{REFERENCE_EMOTION}'))"
    )

    formulas = {
        "Null": (
            f"{analysis_column} ~ 1"
        ),
        "Original emotion": (
            f"{analysis_column} ~ {original_term}"
        ),
        "Reported emotion": (
            f"{analysis_column} ~ {response_term}"
        ),
        "Combined": (
            f"{analysis_column} ~ "
            f"{original_term} + {response_term}"
        ),
    }

    results = {}

    print("\n" + "=" * 80)
    print(f"Metric: {metric}")
    print(f"Observations: {len(data)}")
    print("=" * 80)

    for model_name, formula in formulas.items():
        label = f"{metric} | {model_name}"

        print(f"\nFitting: {label}")
        print(f"Formula: {formula}")

        results[model_name] = fit_mixed_model(
            formula=formula,
            data=data,
            model_name=label,
        )

    comparisons = pd.DataFrame(
        [
            likelihood_ratio_test(
                results["Null"],
                results["Original emotion"],
                metric,
                "Null",
                "Original emotion",
            ),
            likelihood_ratio_test(
                results["Null"],
                results["Reported emotion"],
                metric,
                "Null",
                "Reported emotion",
            ),
            likelihood_ratio_test(
                results["Original emotion"],
                results["Combined"],
                metric,
                "Original emotion",
                "Combined",
            ),
            likelihood_ratio_test(
                results["Reported emotion"],
                results["Combined"],
                metric,
                "Reported emotion",
                "Combined",
            ),
        ]
    )

    fixed_effects = pd.concat(
        [
            extract_fixed_effects(
                result,
                metric,
                model_name,
            )
            for model_name, result in results.items()
        ],
        ignore_index=True,
    )

    pairwise = pd.concat(
        [
            pairwise_emotion_contrasts(
                results["Original emotion"],
                metric,
                "Original emotion",
                "original_emotion",
            ),
            pairwise_emotion_contrasts(
                results["Reported emotion"],
                metric,
                "Reported emotion",
                "response_emotion",
            ),
            pairwise_emotion_contrasts(
                results["Combined"],
                metric,
                "Combined",
                "original_emotion",
            ),
            pairwise_emotion_contrasts(
                results["Combined"],
                metric,
                "Combined",
                "response_emotion",
            ),
        ],
        ignore_index=True,
    )

    descriptive = pd.concat(
        [
            descriptive_summary(
                data,
                metric,
                "original_emotion",
            ),
            descriptive_summary(
                data,
                metric,
                "response_emotion",
            ),
        ],
        ignore_index=True,
    )

    model_information = pd.DataFrame(
        [
            {
                "metric": metric,
                "model": name,
                "formula": formulas[name],
                "n_observations": result.nobs,
                "n_participants": data["participant"].nunique(),
                "n_images": data["filename"].nunique(),
                "log_likelihood": result.llf,
                "aic": result.aic,
                "bic": result.bic,
                "converged": getattr(
                    result,
                    "converged",
                    np.nan,
                ),
            }
            for name, result in results.items()
        ]
    )

    for name, result in results.items():
        summary_file.write("=" * 100 + "\n")
        summary_file.write(f"Metric: {metric}\n")
        summary_file.write(f"Model: {name}\n")
        summary_file.write(f"Formula: {formulas[name]}\n")
        summary_file.write("=" * 100 + "\n")
        summary_file.write(result.summary().as_text())
        summary_file.write("\n\n")

    return {
        "data": data,
        "model_information": model_information,
        "comparisons": comparisons,
        "fixed_effects": fixed_effects,
        "pairwise": pairwise,
        "descriptive": descriptive,
    }


# ==========================================================
# MAIN
# ==========================================================

def main() -> None:
    data, resolved_columns = load_data()

    print("\nDataset summary")
    print("----------------")
    print(f"Rows loaded: {len(data)}")
    print(
        f"Participants: {data['participant'].nunique()}"
    )
    print(f"Images: {data['filename'].nunique()}")

    print("\nResolved metric columns:")
    for canonical, source in resolved_columns.items():
        print(f"  {canonical} <- {source}")

    all_model_information = []
    all_comparisons = []
    all_fixed_effects = []
    all_pairwise = []
    all_descriptive = []
    analysis_datasets = {}

    with open(
        SUMMARY_TXT,
        "w",
        encoding="utf-8",
    ) as summary_file:
        for metric, config in METRICS.items():
            outputs = analyze_metric(
                full_data=data,
                metric=metric,
                analysis_column=config["analysis_column"],
                summary_file=summary_file,
            )

            analysis_datasets[metric] = outputs["data"]
            all_model_information.append(
                outputs["model_information"]
            )
            all_comparisons.append(
                outputs["comparisons"]
            )
            all_fixed_effects.append(
                outputs["fixed_effects"]
            )
            all_pairwise.append(
                outputs["pairwise"]
            )
            all_descriptive.append(
                outputs["descriptive"]
            )

    model_information = pd.concat(
        all_model_information,
        ignore_index=True,
    )

    model_comparisons = pd.concat(
        all_comparisons,
        ignore_index=True,
    )

    fixed_effects = pd.concat(
        all_fixed_effects,
        ignore_index=True,
    )

    pairwise = pd.concat(
        all_pairwise,
        ignore_index=True,
    )

    descriptive = pd.concat(
        all_descriptive,
        ignore_index=True,
    )

    model_information.to_csv(
        MODEL_INFO_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    model_comparisons.to_csv(
        MODEL_COMPARISONS_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    fixed_effects.to_csv(
        FIXED_EFFECTS_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    pairwise.to_csv(
        PAIRWISE_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    descriptive.to_csv(
        DESCRIPTIVE_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    with pd.ExcelWriter(
        OUTPUT_XLSX,
        engine="openpyxl",
    ) as writer:
        model_information.to_excel(
            writer,
            sheet_name="Model Information",
            index=False,
        )

        model_comparisons.to_excel(
            writer,
            sheet_name="Model Comparisons",
            index=False,
        )

        fixed_effects.to_excel(
            writer,
            sheet_name="Fixed Effects",
            index=False,
        )

        pairwise.to_excel(
            writer,
            sheet_name="Pairwise Contrasts",
            index=False,
        )

        descriptive.to_excel(
            writer,
            sheet_name="Descriptive Statistics",
            index=False,
        )

        # Excel sheet names are limited to 31 characters.
        dataset_sheet_names = {
            "dwell_time_s": "Data Dwell Time",
            "fixation_count": "Data Fixation Count",
            "mean_fixation_duration_ms": "Data Mean Fix Duration",
            "fixation_density_per_megapixel": "Data Fixation Density",
        }

        for metric, metric_data in analysis_datasets.items():
            metric_data.to_excel(
                writer,
                sheet_name=dataset_sheet_names[metric],
                index=False,
            )

    print("\nGlobal model comparisons")
    print("------------------------")
    print(
        model_comparisons[
            [
                "metric",
                "reduced_model",
                "full_model",
                "lr_statistic",
                "df_difference",
                "p_value",
                "significant_0_05",
            ]
        ].to_string(index=False)
    )

    print("\nDone.")
    print(f"Results workbook: {OUTPUT_XLSX}")
    print(f"Model summaries:  {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
