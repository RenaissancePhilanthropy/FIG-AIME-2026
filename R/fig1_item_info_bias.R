#' Experiment 1: Metrics are Biased/Polluted by Item Info
#'
#' This experiment demonstrates that traditional metrics are polluted by item
#' difficulty information. We compare:
#' - Random baseline vs actual model performance
#' - With 0 student info vs with lots of student info
#'
#' Expected finding: FIG properly accounts for item difficulty, showing that
#' a model predicting only item difficulty has FIG ~ 0, while a model that
#' captures student ability has FIG > 0.
#'
#' Paper Figure 1.

library(tidyverse)
library(cowplot)
library(figmetric)

# Resolve the repo root from this script's path (run via Rscript, e.g. through the Makefile).
.file <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
repo_root <- dirname(dirname(normalizePath(.file)))
fig_dir    <- file.path(repo_root, "figures")

set.seed(42)

# ==============================================================================
# Simulation Parameters
# ==============================================================================

# A shared 200-item pool, with SEPARATE train and eval student populations.
# Responses-per-item in training = n_train_students * items_per_student / n_items
#   = 2000 * 20 / 200 = 200, still enough to estimate item base rates precisely so
# the item-only FIG sits at ~0 (the estimation-gap bias scales ~1/responses-per-item).
# Only 50 eval students, so the student-bootstrap CIs (error bars) are wide.
n_items <- 200
n_train_students <- 2000
n_eval_students <- 50
items_per_student <- 20           # distinct items each student answers
discriminability <- 1

# Item difficulties (logit scale) and per-population abilities (theta ~ N(0,1)).
# NOTE: the first (runif) draw is intentionally discarded. It advances the RNG stream so the
# generated data -- and hence the figure -- matches the published one; do not remove it.
item_difficulties <- runif(n_items, min = -2, max = 2)
item_difficulties <- rnorm(n_items, 0, 1)
train_abilities <- rnorm(n_train_students, mean = 0, sd = 1)
eval_abilities  <- rnorm(n_eval_students,  mean = 0, sd = 1)

# ==============================================================================
# Generate two clean response sets (train and eval)
# ==============================================================================
# Each student answers `items_per_student` DISTINCT items drawn from the pool;
# correctness is IRT-generated from the student's ability and item difficulty.
make_responses <- function(abilities) {
  ns <- length(abilities)
  student_id <- rep(seq_len(ns), each = items_per_student)
  item_id <- as.vector(vapply(
    seq_len(ns), function(s) sample.int(n_items, items_per_student), integer(items_per_student)
  ))
  true_prob <- 1 / (1 + exp(-discriminability * (abilities[student_id] - item_difficulties[item_id])))
  tibble(student_id, item_id, true_prob, y = rbinom(length(true_prob), 1, true_prob))
}

train_data <- make_responses(train_abilities)   # 2000 x 20 = 40k responses (~200/item)
eval_data  <- make_responses(eval_abilities)    # 50 x 20 = 1000 responses

# ==============================================================================
# Model 1: Random Baseline (no information)
# ==============================================================================

# Predicts global mean for everything
global_mean <- mean(train_data$y)
eval_data$pred_random <- global_mean

# ==============================================================================
# Model 2: Item-Only Model (predicts only item difficulty)
# ==============================================================================

# This model has access to item difficulty but not student ability
# It predicts the empirical item mean from training data
item_means <- train_data %>%
  group_by(item_id) %>%
  summarise(item_mean = mean(y), .groups = "drop")

eval_data <- eval_data %>%
  left_join(item_means, by = "item_id") %>%
  mutate(pred_item_only = coalesce(item_mean, global_mean))

# ==============================================================================
# Model 3: Student-Aware Model (knows student ability)
# ==============================================================================
#
# Simulates a model that has learned student abilities from history
# We give it the true IRT probability (best case scenario)

eval_data$pred_student_aware <- eval_data$true_prob

# ==============================================================================
# Compute Traditional Metrics
# ==============================================================================

compute_accuracy <- function(y_pred, y_true) {
  mean((y_pred > 0.5) == y_true)
}

compute_auc <- function(y_pred, y_true) {
  as.numeric(pROC::auc(y_true, y_pred))
}

# ==============================================================================
# Compute FIG for Each Model
# ==============================================================================

compute_all_metrics <- function(predictions, name) {
  # Traditional metrics
  acc <- compute_accuracy(predictions, eval_data$y)
  auc <- compute_auc(predictions, eval_data$y)

  # FIG (accounts for item difficulty)
  fig_result <- fractional_information_gain_validation(
    y_pred_eval = predictions,
    y_eval = eval_data$y,
    item_id_eval = eval_data$item_id,
    student_id_eval = eval_data$student_id,
    y_train = train_data$y,
    item_id_train = train_data$item_id,
    use_shrinkage = TRUE
  )

  tibble(
    model = name,
    accuracy = acc,
    auc = auc,
    fig_v = fig_result$fig_v,
    fig_v_pooled = fig_result$fig_v_pooled
  )
}

results <- bind_rows(
  compute_all_metrics(eval_data$pred_random, "Random Baseline"),
  compute_all_metrics(eval_data$pred_item_only, "Item-Only Model"),
  compute_all_metrics(eval_data$pred_student_aware, "Bayes-Optimal")
)

# The metric values behind the plot, to compare a run against the paper.
cat("\nMetrics by model:\n")
print(results %>%
        mutate(across(where(is.numeric), ~round(., 4))) %>%
        as.data.frame())
cat("\n")

# ==============================================================================
# Student-level bootstrap CIs for each metric
# ==============================================================================
# Resample whole EVAL students with replacement (the independent unit) and
# recompute Accuracy, AUC and FIG on each resample, calling the FIG library itself
# (fractional_information_gain_validation) so the metric is never reimplemented.
# Each resampled student gets a fresh id so FIG's per-student average counts
# duplicates separately. Replicates run in parallel with mcmapply (fork-based, so the
# workers inherit train_data/eval_data and the loaded packages for free). L'Ecuyer-CMRG
# streams are assigned per worker, so whisker lengths vary slightly with the core
# count; the seeded bar heights are unaffected. Bump n_boot for smoother intervals.

library(parallel)

rows_by_student <- split(seq_len(nrow(eval_data)), eval_data$student_id)
n_stu   <- length(rows_by_student)
n_boot  <- 500
n_cores <- max(1L, detectCores() - 1L)

bootstrap_ci <- function(pred_col) {
  preds <- eval_data[[pred_col]]; y <- eval_data$y; item <- eval_data$item_id
  one_rep <- function(b) {
    samp <- sample.int(n_stu, n_stu, replace = TRUE)
    idx_list <- rows_by_student[samp]
    idx <- unlist(idx_list, use.names = FALSE)
    new_sid <- rep.int(seq_along(samp), lengths(idx_list))  # fresh per-student ids
    yb <- y[idx]; pb <- preds[idx]; ib <- item[idx]
    fr <- fractional_information_gain_validation(
      y_pred_eval = pb, y_eval = yb, item_id_eval = ib, student_id_eval = new_sid,
      y_train = train_data$y, item_id_train = train_data$item_id, use_shrinkage = TRUE
    )
    c(acc = mean((pb > 0.5) == yb),
      auc = if (length(unique(yb)) > 1) as.numeric(pROC::auc(yb, pb, quiet = TRUE)) else NA_real_,
      fig = fr$fig_v)
  }
  res <- mcmapply(one_rep, seq_len(n_boot), mc.cores = n_cores)  # 3 x n_boot matrix
  qq <- function(x) as.numeric(quantile(x, c(0.025, 0.975), na.rm = TRUE))
  tibble(
    accuracy_lo = qq(res["acc", ])[1], accuracy_hi = qq(res["acc", ])[2],
    auc_lo      = qq(res["auc", ])[1], auc_hi      = qq(res["auc", ])[2],
    fig_v_lo    = qq(res["fig", ])[1], fig_v_hi    = qq(res["fig", ])[2]
  )
}

RNGkind("L'Ecuyer-CMRG")   # parallel-safe, reproducible streams for mcmapply
set.seed(123)
ci_long <- bind_rows(
  bootstrap_ci("pred_random")        %>% mutate(model = "Random Baseline"),
  bootstrap_ci("pred_item_only")     %>% mutate(model = "Item-Only Model"),
  bootstrap_ci("pred_student_aware") %>% mutate(model = "Bayes-Optimal")
) %>%
  pivot_longer(-model, names_to = c("metric", "bound"), names_pattern = "(.*)_(lo|hi)") %>%
  pivot_wider(names_from = bound, values_from = value) %>%
  mutate(metric = factor(recode(metric, accuracy = "Accuracy", auc = "AUC", fig_v = "FIG-V"),
                         levels = c("Accuracy", "AUC", "FIG-V")))

# ==============================================================================
# Visualization
# ==============================================================================

# Prepare data for plotting
plot_data <- results %>%
  pivot_longer(cols = c(accuracy, auc, fig_v),
               names_to = "metric",
               values_to = "value") %>%
  bind_rows(tibble(
    model='Oracle',
    metric=c('accuracy','auc','fig_v'),
    value=c(1, 1, 1)
  )) %>%
  mutate(
    metric = factor(metric,
                    levels = c("accuracy", "auc", "fig_v"),
                    labels = c("Accuracy", "AUC", "FIG-V")),
    model = factor(model, levels = c("Random Baseline", "Item-Only Model",
                                     "Bayes-Optimal", "Oracle"))
  )

# Attach the bootstrap CIs (Oracle reference has none -> NA, so no error bar).
plot_data <- plot_data %>%
  left_join(ci_long %>% mutate(model = factor(model, levels = levels(plot_data$model))),
            by = c("model", "metric"))


p <- plot_data %>%
  ggplot(aes(x = model, y = value, fill = model)) +
  geom_col(width = 0.7) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  geom_errorbar(
    data = function(d) d[!is.na(d$lo), ],
    aes(ymin = lo, ymax = hi),
    width = 0.25, linewidth = 0.4, colour = "gray15"
  ) +
  facet_wrap(~metric, ncol = 3) +
  scale_fill_manual(values = c(
    "Random Baseline" = "#E74C3C",
    "Item-Only Model" = "#F39C12",
    "Bayes-Optimal" = "#27AE60",
    'Oracle' = '#444555'
  )) +
  labs(
    title = "FIG-V isolates student information",
    x = NULL,
    y = "Metric Value"
  ) +
  theme_cowplot(font_size = 14) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 12),
    legend.position = "none"
  ) +
  scale_y_continuous(limits=c(NA,1), breaks=seq(-.25, 1, by=0.25))

# Save plot
out_path <- file.path(fig_dir, "01_item_info_bias_plot.pdf")
ggsave(out_path, p, width = 6, height = 3.85, dpi = 300, bg = "white")
cat("Plot saved to:", out_path, "\n")
