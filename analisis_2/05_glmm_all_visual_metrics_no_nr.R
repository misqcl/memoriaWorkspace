# glmm_all_visual_metrics.R
#
# Distribution-appropriate crossed generalized mixed-effects models
# for the four visual metrics used.
#
# Models:
#   fixation_count                    -> Negative-binomial GLMM
#   dwell_time_s                      -> Gamma GLMM with log link
#   mean_fixation_duration_ms         -> Gamma GLMM with log link
#   fixation_density_per_megapixel    -> Gamma GLMM with log link
#
# Fixed effects tested for every metric:
#   1. Null model
#   2. Original emotion
#   3. Reported emotion
#   4. Combined original + reported emotion
#
# Random intercepts:
#   Primary: (1 | participant) + (1 | filename)
#   Automatic fallback for an entire metric when any crossed model is invalid:
#   (1 | participant)
#
# Input:
#   memoria/analisis_2/heatmap_analysis/heatmap_analysis_clean_no_nr.xlsx
#   sheet: User Image Metrics
#
# Output:
#   memoria/glmm_all_visual_metrics/
#       glmm_all_metrics_results.xlsx
#       model_information.csv
#       model_comparisons.csv
#       fixed_effects.csv
#       pairwise_contrasts.csv
#       descriptive_statistics.csv
#       model_summaries.txt
#
# Install once in R:
# install.packages(c(
#   "readxl", "dplyr", "tidyr", "stringr",
#   "glmmTMB", "emmeans", "openxlsx", "broom.mixed",
#   "performance"
# ))

suppressPackageStartupMessages({
  library(readxl)
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(glmmTMB)
  library(emmeans)
  library(openxlsx)
  library(broom.mixed)
  library(performance)
})

# ============================================================
# PATHS
# ============================================================

script_path <- tryCatch(
  normalizePath(sys.frame(1)$ofile),
  error = function(e) NA_character_
)

if (is.na(script_path)) {
  script_dir <- getwd()
} else {
  script_dir <- dirname(script_path)
}

base_dir <- dirname(script_dir)

input_file <- file.path(
  base_dir,
  "analisis_2",
  "heatmap_analysis",
  "heatmap_analysis_clean_no_nr.xlsx"
)

input_sheet <- "User Image Metrics"

output_dir <- file.path(
  base_dir,
  "analisis_2",
  "glmm_all_visual_metrics"
)

dir.create(
  output_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

output_xlsx <- file.path(
  output_dir,
  "glmm_all_metrics_results.xlsx"
)

model_info_csv <- file.path(
  output_dir,
  "model_information.csv"
)

model_comparisons_csv <- file.path(
  output_dir,
  "model_comparisons.csv"
)

fixed_effects_csv <- file.path(
  output_dir,
  "fixed_effects.csv"
)

pairwise_csv <- file.path(
  output_dir,
  "pairwise_contrasts.csv"
)

descriptive_csv <- file.path(
  output_dir,
  "descriptive_statistics.csv"
)

summary_txt <- file.path(
  output_dir,
  "model_summaries.txt"
)

# ============================================================
# CONFIGURATION
# ============================================================

reference_emotion <- "Neutral"

emotion_levels <- c(
  "Neutral",
  "Negative",
  "Positive"
)

metric_specs <- list(
  fixation_count = list(
    display_name = "Fixation count",
    family = nbinom2(link = "log"),
    distribution = "Negative binomial"
  ),
  dwell_time_s = list(
    display_name = "Dwell time (s)",
    family = Gamma(link = "log"),
    distribution = "Gamma"
  ),
  mean_fixation_duration_ms = list(
    display_name = "Mean fixation duration (ms)",
    family = Gamma(link = "log"),
    distribution = "Gamma"
  ),
  fixation_density_per_megapixel = list(
    display_name = "Fixation density per megapixel",
    family = Gamma(link = "log"),
    distribution = "Gamma"
  )
)

# ============================================================
# HELPERS
# ============================================================

normalize_emotion <- function(x) {
  x_chr <- str_to_lower(str_trim(as.character(x)))

  case_when(
    x_chr %in% c("negative", "negativo", "negativa") ~ "Negative",
    x_chr %in% c("neutral", "neutro", "neutra") ~ "Neutral",
    x_chr %in% c("positive", "positivo", "positiva") ~ "Positive",
    x_chr %in% c("nr", "n/r", "no responde", "no response", "no_response") ~ NA_character_,
    TRUE ~ str_trim(as.character(x))
  )
}

resolve_column <- function(data, candidates, canonical_name) {
  exact <- candidates[candidates %in% names(data)]

  if (length(exact) > 0) {
    return(exact[[1]])
  }

  lower_names <- setNames(
    names(data),
    str_to_lower(names(data))
  )

  lower_candidates <- str_to_lower(candidates)
  matched <- lower_candidates[
    lower_candidates %in% names(lower_names)
  ]

  if (length(matched) > 0) {
    return(lower_names[[matched[[1]]]])
  }

  stop(
    paste0(
      "Could not find a column for '",
      canonical_name,
      "'. Tried: ",
      paste(candidates, collapse = ", "),
      ". Available columns: ",
      paste(names(data), collapse = ", ")
    )
  )
}

resolve_metric_columns <- function(data) {
  list(
    dwell_time_s = resolve_column(
      data,
      c(
        "dwell_time_s",
        "total_dwell_time_s",
        "dwell_time"
      ),
      "dwell_time_s"
    ),
    fixation_count = resolve_column(
      data,
      c(
        "fixation_count",
        "total_fixations",
        "number_of_fixations"
      ),
      "fixation_count"
    ),
    mean_fixation_duration_ms = resolve_column(
      data,
      c(
        "mean_fixation_duration_ms",
        "average_fixation_duration_ms",
        "avg_fixation_duration_ms"
      ),
      "mean_fixation_duration_ms"
    ),
    fixation_density_per_megapixel = resolve_column(
      data,
      c(
        "fixation_density_per_megapixel",
        "mean_fixation_density_per_megapixel",
        "fixation_density"
      ),
      "fixation_density_per_megapixel"
    )
  )
}

load_analysis_data <- function() {
  if (!file.exists(input_file)) {
    stop(
      paste0(
        "Input file not found: ",
        input_file
      )
    )
  }

  data <- read_excel(
    input_file,
    sheet = input_sheet
  )

  required <- c(
    "participant",
    "filename",
    "original_emotion",
    "response_emotion"
  )

  missing <- setdiff(
    required,
    names(data)
  )

  if (length(missing) > 0) {
    stop(
      paste0(
        "Missing required columns: ",
        paste(missing, collapse = ", ")
      )
    )
  }

  metric_columns <- resolve_metric_columns(data)

  selected_columns <- c(
    required,
    unlist(metric_columns)
  )

  data <- data %>%
    select(all_of(selected_columns))

  rename_pairs <- setNames(
    names(metric_columns),
    unlist(metric_columns)
  )

  for (source_name in names(rename_pairs)) {
    canonical_name <- rename_pairs[[source_name]]

    if (source_name != canonical_name) {
      names(data)[names(data) == source_name] <- canonical_name
    }
  }

  data <- data %>%
    mutate(
      participant = factor(as.character(participant)),
      filename = factor(as.character(filename)),
      original_emotion = normalize_emotion(original_emotion),
      response_emotion = normalize_emotion(response_emotion)
    ) %>%
    filter(
      original_emotion %in% c("Negative", "Neutral", "Positive"),
      response_emotion %in% c("Negative", "Neutral", "Positive")
    ) %>%
    mutate(
      original_emotion = factor(
        original_emotion,
        levels = emotion_levels
      ),
      response_emotion = factor(
        response_emotion,
        levels = emotion_levels
      ),
      across(
        all_of(names(metric_specs)),
        ~ suppressWarnings(as.numeric(.x))
      )
    )

  attr(data, "metric_columns") <- metric_columns

  data
}

is_model_valid <- function(model) {
  if (is.null(model)) {
    return(FALSE)
  }

  pd_hessian <- tryCatch(
    isTRUE(model$sdr$pdHess),
    error = function(e) FALSE
  )

  log_likelihood <- tryCatch(
    as.numeric(logLik(model)),
    error = function(e) NA_real_
  )

  information_criteria <- tryCatch(
    c(AIC(model), BIC(model)),
    error = function(e) c(NA_real_, NA_real_)
  )

  coefficient_table <- tryCatch(
    summary(model)$coefficients$cond,
    error = function(e) NULL
  )

  finite_coefficients <- !is.null(coefficient_table) &&
    all(is.finite(coefficient_table[, "Estimate"])) &&
    all(is.finite(coefficient_table[, "Std. Error"]))

  isTRUE(pd_hessian) &&
    is.finite(log_likelihood) &&
    all(is.finite(information_criteria)) &&
    finite_coefficients
}

fit_glmm <- function(
  formula,
  data,
  family,
  model_label
) {
  attempts <- list(
    list(
      name = "nlminb",
      control = glmmTMBControl(
        optimizer = nlminb,
        optCtrl = list(
          iter.max = 10000,
          eval.max = 10000
        )
      )
    ),
    list(
      name = "BFGS",
      control = glmmTMBControl(
        optimizer = optim,
        optArgs = list(
          method = "BFGS"
        ),
        optCtrl = list(
          maxit = 10000
        )
      )
    )
  )

  fitted_models <- list()

  for (attempt in attempts) {
    model <- tryCatch(
      suppressWarnings(
        glmmTMB(
          formula = formula,
          data = data,
          family = family,
          REML = FALSE,
          control = attempt$control
        )
      ),
      error = function(e) {
        message(
          model_label,
          ": ",
          attempt$name,
          " failed: ",
          conditionMessage(e)
        )
        NULL
      }
    )

    if (is.null(model)) {
      next
    }

    pd_hessian <- tryCatch(
      isTRUE(model$sdr$pdHess),
      error = function(e) FALSE
    )

    fitted_models[[attempt$name]] <- model

    message(
      model_label,
      ": fitted with ",
      attempt$name,
      "; positive-definite Hessian=",
      pd_hessian
    )

    valid_model <- is_model_valid(model)

    message(
      model_label,
      "; finite likelihood and standard errors=",
      valid_model
    )

    if (valid_model) {
      return(model)
    }
  }

  if (length(fitted_models) > 0) {
    warning(
      paste0(
        model_label,
        ": no optimizer produced a valid model with a positive-definite ",
        "Hessian and finite likelihood/standard errors."
      )
    )
    return(NULL)
  }

  warning(
    paste0(
      "All optimizers failed for ",
      model_label,
      "."
    )
  )
  NULL
}

safe_lrt <- function(
  reduced_model,
  full_model,
  metric,
  reduced_name,
  full_name
) {
  empty_result <- tibble(
    metric = metric,
    reduced_model = reduced_name,
    full_model = full_name,
    lr_statistic = NA_real_,
    df_difference = NA_real_,
    p_value = NA_real_,
    significant_0_05 = NA
  )

  if (!is_model_valid(reduced_model) || !is_model_valid(full_model)) {
    warning(
      paste0(
        metric,
        " | ", reduced_name, " vs ", full_name,
        ": likelihood-ratio comparison skipped because one model is invalid."
      )
    )
    return(empty_result)
  }

  tryCatch({
    comparison <- anova(
      reduced_model,
      full_model
    )

    lr_statistic <- comparison$Chisq[2]
    df_difference <- comparison$`Chi Df`[2]
    p_value <- comparison$`Pr(>Chisq)`[2]

    tibble(
      metric = metric,
      reduced_model = reduced_name,
      full_model = full_name,
      lr_statistic = lr_statistic,
      df_difference = df_difference,
      p_value = p_value,
      significant_0_05 = ifelse(
        is.na(p_value),
        NA,
        p_value < 0.05
      )
    )
  }, error = function(e) {
    warning(
      paste0(
        metric,
        " | ", reduced_name, " vs ", full_name,
        ": likelihood-ratio comparison failed: ",
        conditionMessage(e)
      )
    )
    empty_result
  })
}

extract_model_information <- function(
  fitted_model,
  metric,
  model_name,
  formula_text,
  distribution_name,
  analysis_data,
  random_effect_structure,
  fallback_used
) {
  pd_hessian <- tryCatch(
    fitted_model$sdr$pdHess,
    error = function(e) NA
  )

  singular <- tryCatch(
    check_singularity(fitted_model)$singular,
    error = function(e) NA
  )

  model_log_likelihood <- tryCatch(
    as.numeric(logLik(fitted_model)),
    error = function(e) NA_real_
  )

  model_aic <- tryCatch(
    AIC(fitted_model),
    error = function(e) NA_real_
  )

  model_bic <- tryCatch(
    BIC(fitted_model),
    error = function(e) NA_real_
  )

  tibble(
    metric = metric,
    model = model_name,
    formula = formula_text,
    distribution = distribution_name,
    random_effect_structure = random_effect_structure,
    fallback_used = fallback_used,
    n_observations = nrow(analysis_data),
    n_participants = n_distinct(analysis_data$participant),
    n_images = n_distinct(analysis_data$filename),
    log_likelihood = model_log_likelihood,
    aic = model_aic,
    bic = model_bic,
    positive_definite_hessian = pd_hessian,
    singular_fit = singular
  )
}

extract_fixed_effects <- function(
  model,
  metric,
  model_name
) {
  result <- tryCatch(
    tidy(
      model,
      effects = "fixed",
      component = "cond",
      conf.int = TRUE
    ),
    error = function(e) {
      warning(
        paste0(
          metric,
          " | ",
          model_name,
          ": fixed effects could not be extracted: ",
          conditionMessage(e)
        )
      )
      tibble()
    }
  )

  if (nrow(result) == 0) {
    return(
      tibble(
        metric = character(),
        model = character(),
        term = character(),
        estimate_log_scale = numeric(),
        standard_error = numeric(),
        z_value = numeric(),
        p_value = numeric(),
        ci_95_lower = numeric(),
        ci_95_upper = numeric(),
        rate_ratio = numeric(),
        approximate_percent_change = numeric()
      )
    )
  }

  result %>%
    transmute(
      metric = metric,
      model = model_name,
      term = term,
      estimate_log_scale = estimate,
      standard_error = std.error,
      z_value = statistic,
      p_value = p.value,
      ci_95_lower = conf.low,
      ci_95_upper = conf.high,
      rate_ratio = exp(estimate),
      approximate_percent_change = (
        exp(estimate) - 1
      ) * 100
    )
}

extract_pairwise <- function(
  model,
  metric,
  model_name,
  predictor
) {
  tryCatch({
    emm_formula <- as.formula(
      paste0(
        "~ ",
        predictor
      )
    )

    emm <- emmeans(
      model,
      specs = emm_formula,
      type = "response"
    )

    contrast_table <- pairs(
      emm,
      adjust = "bonferroni"
    ) %>%
      as.data.frame()

    contrast_table %>%
      transmute(
        metric = metric,
        model = model_name,
        predictor = predictor,
        comparison = contrast,
        ratio = ratio,
        standard_error = SE,
        df = df,
        z_or_t_ratio = z.ratio,
        p_value_bonferroni = p.value,
        approximate_percent_difference = (
          ratio - 1
        ) * 100,
        significant_bonferroni_0_05 = p.value < 0.05
      )
  }, error = function(e) {
    warning(
      paste0(
        metric,
        " | ",
        model_name,
        " | ",
        predictor,
        ": pairwise contrasts unavailable: ",
        conditionMessage(e)
      )
    )

    tibble(
      metric = character(),
      model = character(),
      predictor = character(),
      comparison = character(),
      ratio = numeric(),
      standard_error = numeric(),
      df = numeric(),
      z_or_t_ratio = numeric(),
      p_value_bonferroni = numeric(),
      approximate_percent_difference = numeric(),
      significant_bonferroni_0_05 = logical()
    )
  })
}

descriptive_summary <- function(
  data,
  metric,
  predictor
) {
  data %>%
    group_by(
      .data[[predictor]]
    ) %>%
    summarise(
      observations = n(),
      mean = mean(
        .data[[metric]],
        na.rm = TRUE
      ),
      standard_deviation = sd(
        .data[[metric]],
        na.rm = TRUE
      ),
      median = median(
        .data[[metric]],
        na.rm = TRUE
      ),
      minimum = min(
        .data[[metric]],
        na.rm = TRUE
      ),
      maximum = max(
        .data[[metric]],
        na.rm = TRUE
      ),
      .groups = "drop"
    ) %>%
    rename(
      category = 1
    ) %>%
    mutate(
      metric = metric,
      grouping_variable = predictor,
      .before = 1
    )
}

write_model_summary <- function(
  connection,
  model,
  metric,
  model_name,
  formula_text
) {
  writeLines(
    rep("=", 100),
    connection
  )

  writeLines(
    paste0(
      "Metric: ",
      metric
    ),
    connection
  )

  writeLines(
    paste0(
      "Model: ",
      model_name
    ),
    connection
  )

  writeLines(
    paste0(
      "Formula: ",
      formula_text
    ),
    connection
  )

  writeLines(
    rep("=", 100),
    connection
  )

  capture.output(
    summary(model),
    file = connection
  )

  writeLines(
    c("", ""),
    connection
  )
}

# ============================================================
# ANALYSIS FOR ONE METRIC
# ============================================================

analyze_metric <- function(
  full_data,
  metric,
  spec,
  summary_connection
) {
  analysis_data <- full_data %>%
    filter(
      !is.na(.data[[metric]])
    )

  if (metric == "fixation_count") {
    analysis_data <- analysis_data %>%
      filter(
        .data[[metric]] >= 0,
        abs(
          .data[[metric]] -
          round(.data[[metric]])
        ) < 1e-8
      ) %>%
      mutate(
        fixation_count = as.integer(
          round(fixation_count)
        )
      )
  } else {
    # Gamma models require strictly positive responses.
    analysis_data <- analysis_data %>%
      filter(
        .data[[metric]] > 0
      )
  }

  if (nrow(analysis_data) == 0) {
    stop(
      paste0(
        "No valid observations available for ",
        metric
      )
    )
  }

  model_response <- metric

  if (metric != "fixation_count") {
    positive_median <- median(
      analysis_data[[metric]],
      na.rm = TRUE
    )

    if (!is.finite(positive_median) || positive_median <= 0) {
      positive_median <- 1
    }

    model_response <- paste0(
      metric,
      "_scaled"
    )

    analysis_data[[model_response]] <- (
      analysis_data[[metric]] /
      positive_median
    )
  }

  make_formulas <- function(random_terms) {
    list(
      Null = as.formula(
        paste0(
          model_response,
          " ~ 1 + ",
          random_terms
        )
      ),
      `Original emotion` = as.formula(
        paste0(
          model_response,
          " ~ original_emotion + ",
          random_terms
        )
      ),
      `Reported emotion` = as.formula(
        paste0(
          model_response,
          " ~ response_emotion + ",
          random_terms
        )
      ),
      Combined = as.formula(
        paste0(
          model_response,
          " ~ original_emotion + response_emotion + ",
          random_terms
        )
      )
    )
  }

  fit_model_set <- function(formulas, structure_label, model_family) {
    fitted <- list()

    for (model_name in names(formulas)) {
      model_label <- paste(
        metric,
        model_name,
        structure_label,
        sep = " | "
      )

      # Use single-bracket assignment so NULL models are preserved as
      # named list elements. With [[<- NULL, R removes the element, which
      # would make all(valid_models) incorrectly return TRUE.
      fitted[model_name] <- list(
        fit_glmm(
          formula = formulas[[model_name]],
          data = analysis_data,
          family = model_family,
          model_label = model_label
        )
      )
    }

    fitted
  }

  message("")
  message(
    paste(
      rep("=", 80),
      collapse = ""
    )
  )
  message(
    "Metric: ",
    metric
  )
  message(
    "Observations: ",
    nrow(analysis_data)
  )
  message(
    paste(
      rep("=", 80),
      collapse = ""
    )
  )

  crossed_random_terms <- "(1 | participant) + (1 | filename)"
  participant_random_terms <- "(1 | participant)"

  random_effect_structure <- crossed_random_terms
  fallback_used <- FALSE
  active_family <- spec$family
  active_distribution <- spec$distribution

  formulas <- make_formulas(crossed_random_terms)
  models <- fit_model_set(
    formulas,
    "crossed random intercepts",
    active_family
  )

  valid_models <- vapply(
    models,
    is_model_valid,
    logical(1)
  )

  if (!all(valid_models) && metric == "fixation_count") {
    invalid_names <- names(valid_models)[!valid_models]

    warning(
      paste0(
        metric,
        ": invalid negative-binomial crossed model(s): ",
        paste(invalid_names, collapse = ", "),
        ". Refitting the complete model set as Poisson GLMMs with the same ",
        "crossed random-effects structure. This is appropriate when the ",
        "negative-binomial dispersion approaches the Poisson boundary."
      )
    )

    active_family <- poisson(link = "log")
    active_distribution <- paste0(
      "Poisson (automatic fallback from negative binomial; ",
      "negative-binomial fit reached the Poisson boundary)"
    )
    fallback_used <- TRUE

    models <- fit_model_set(
      formulas,
      "Poisson crossed random intercepts fallback",
      active_family
    )

    valid_models <- vapply(
      models,
      is_model_valid,
      logical(1)
    )
  }

  if (!all(valid_models)) {
    invalid_names <- names(valid_models)[!valid_models]

    warning(
      paste0(
        metric,
        ": invalid crossed model(s): ",
        paste(invalid_names, collapse = ", "),
        ". Refitting the complete model set with participant random intercept only, ",
        "so all likelihood-ratio comparisons retain the same random-effects structure."
      )
    )

    random_effect_structure <- participant_random_terms
    fallback_used <- TRUE
    formulas <- make_formulas(participant_random_terms)
    models <- fit_model_set(
      formulas,
      if (metric == "fixation_count") {
        "Poisson participant random intercept fallback"
      } else {
        "participant random intercept fallback"
      },
      active_family
    )

    fallback_valid <- vapply(
      models,
      is_model_valid,
      logical(1)
    )

    if (!all(fallback_valid)) {
      stop(
        paste0(
          metric,
          ": the final fallback model set still contains invalid models: ",
          paste(names(fallback_valid)[!fallback_valid], collapse = ", "),
          ". Results were not written for this metric."
        )
      )
    }
  }

  comparisons <- bind_rows(
    safe_lrt(
      models$Null,
      models$`Original emotion`,
      metric,
      "Null",
      "Original emotion"
    ),
    safe_lrt(
      models$Null,
      models$`Reported emotion`,
      metric,
      "Null",
      "Reported emotion"
    ),
    safe_lrt(
      models$`Original emotion`,
      models$Combined,
      metric,
      "Original emotion",
      "Combined"
    ),
    safe_lrt(
      models$`Reported emotion`,
      models$Combined,
      metric,
      "Reported emotion",
      "Combined"
    )
  )

  fixed_effects <- bind_rows(
    lapply(
      names(models),
      function(model_name) {
        extract_fixed_effects(
          models[[model_name]],
          metric,
          model_name
        )
      }
    )
  )

  pairwise <- bind_rows(
    extract_pairwise(
      models$`Original emotion`,
      metric,
      "Original emotion",
      "original_emotion"
    ),
    extract_pairwise(
      models$`Reported emotion`,
      metric,
      "Reported emotion",
      "response_emotion"
    ),
    extract_pairwise(
      models$Combined,
      metric,
      "Combined",
      "original_emotion"
    ),
    extract_pairwise(
      models$Combined,
      metric,
      "Combined",
      "response_emotion"
    )
  )

  descriptive <- bind_rows(
    descriptive_summary(
      analysis_data,
      metric,
      "original_emotion"
    ),
    descriptive_summary(
      analysis_data,
      metric,
      "response_emotion"
    )
  )

  model_information <- bind_rows(
    lapply(
      names(models),
      function(model_name) {
        extract_model_information(
          fitted_model = models[[model_name]],
          metric = metric,
          model_name = model_name,
          formula_text = paste(
            deparse(
              formulas[[model_name]]
            ),
            collapse = ""
          ),
          distribution_name = active_distribution,
          analysis_data = analysis_data,
          random_effect_structure = random_effect_structure,
          fallback_used = fallback_used
        )
      }
    )
  )

  for (model_name in names(models)) {
    write_model_summary(
      connection = summary_connection,
      model = models[[model_name]],
      metric = metric,
      model_name = model_name,
      formula_text = paste(
        deparse(
          formulas[[model_name]]
        ),
        collapse = ""
      )
    )
  }

  list(
    data = analysis_data,
    model_information = model_information,
    comparisons = comparisons,
    fixed_effects = fixed_effects,
    pairwise = pairwise,
    descriptive = descriptive
  )
}

# ============================================================
# MAIN
# ============================================================

main <- function() {
  data <- load_analysis_data()

  resolved_columns <- attr(
    data,
    "metric_columns"
  )

  message("")
  message("Dataset summary")
  message("----------------")
  message(
    "Rows loaded: ",
    nrow(data)
  )
  message(
    "Participants: ",
    n_distinct(data$participant)
  )
  message(
    "Images: ",
    n_distinct(data$filename)
  )

  message("")
  message("Resolved metric columns:")

  for (metric_name in names(resolved_columns)) {
    message(
      "  ",
      metric_name,
      " <- ",
      resolved_columns[[metric_name]]
    )
  }

  summary_connection <- file(
    summary_txt,
    open = "wt",
    encoding = "UTF-8"
  )

  on.exit(
    close(summary_connection),
    add = TRUE
  )

  outputs <- list()

  for (metric in names(metric_specs)) {
    outputs[[metric]] <- analyze_metric(
      full_data = data,
      metric = metric,
      spec = metric_specs[[metric]],
      summary_connection = summary_connection
    )
  }

  model_information <- bind_rows(
    lapply(
      outputs,
      `[[`,
      "model_information"
    )
  )

  model_comparisons <- bind_rows(
    lapply(
      outputs,
      `[[`,
      "comparisons"
    )
  )

  fixed_effects <- bind_rows(
    lapply(
      outputs,
      `[[`,
      "fixed_effects"
    )
  )

  pairwise <- bind_rows(
    lapply(
      outputs,
      `[[`,
      "pairwise"
    )
  )

  descriptive <- bind_rows(
    lapply(
      outputs,
      `[[`,
      "descriptive"
    )
  )

  write.csv(
    model_information,
    model_info_csv,
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )

  write.csv(
    model_comparisons,
    model_comparisons_csv,
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )

  write.csv(
    fixed_effects,
    fixed_effects_csv,
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )

  write.csv(
    pairwise,
    pairwise_csv,
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )

  write.csv(
    descriptive,
    descriptive_csv,
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )

  workbook <- createWorkbook()

  addWorksheet(
    workbook,
    "Model Information"
  )
  writeData(
    workbook,
    "Model Information",
    model_information
  )

  addWorksheet(
    workbook,
    "Model Comparisons"
  )
  writeData(
    workbook,
    "Model Comparisons",
    model_comparisons
  )

  addWorksheet(
    workbook,
    "Fixed Effects"
  )
  writeData(
    workbook,
    "Fixed Effects",
    fixed_effects
  )

  addWorksheet(
    workbook,
    "Pairwise Contrasts"
  )
  writeData(
    workbook,
    "Pairwise Contrasts",
    pairwise
  )

  addWorksheet(
    workbook,
    "Descriptive Statistics"
  )
  writeData(
    workbook,
    "Descriptive Statistics",
    descriptive
  )

  dataset_sheet_names <- c(
    fixation_count = "Data Fixation Count",
    dwell_time_s = "Data Dwell Time",
    mean_fixation_duration_ms = "Data Mean Fix Duration",
    fixation_density_per_megapixel = "Data Fixation Density"
  )

  for (metric in names(outputs)) {
    sheet_name <- dataset_sheet_names[[metric]]

    addWorksheet(
      workbook,
      sheet_name
    )

    writeData(
      workbook,
      sheet_name,
      outputs[[metric]]$data
    )
  }

  saveWorkbook(
    workbook,
    output_xlsx,
    overwrite = TRUE
  )

  message("")
  message("Global model comparisons")
  message("------------------------")

  print(
    model_comparisons %>%
      select(
        metric,
        reduced_model,
        full_model,
        lr_statistic,
        df_difference,
        p_value,
        significant_0_05
      )
  )

  message("")
  message("Done.")
  message(
    "Results workbook: ",
    output_xlsx
  )
  message(
    "Model summaries: ",
    summary_txt
  )
}

main()