#' Experiment 10b: the two axes of metric sensitivity (ability error vs calibration)
#'
#' Small-multiples grid on the same synthetic IRT setup (true model a=1, theta=0;
#' predictions = sigma(a*(theta_hat - d)); 5000 MC response draws): one row per
#' metric, two columns:
#'   A (left): metrics vs ability-estimate error (theta - theta_hat) at correct
#'      calibration (a = 1).
#'   B (right): metrics vs the discrimination/calibration parameter a (0.25..4,
#'      log2 x-axis) at correct ability (theta_hat = theta).
#'
#' Punchline: accuracy/AUC/F1 are (near-)flat against calibration (column B) while
#' FIG-V and RMSE respond; FIG-V degrades under both miscalibration (column B) and
#' ability error (column A).

library(tidyverse)
library(cowplot)
library(parallel)   # mcmapply: parallel Monte Carlo over the metric sweeps
library(figmetric)

# Paper Figure 3.

# Resolve the repo root from this script's path (run via Rscript, e.g. through the Makefile).
.file <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
repo_root <- dirname(dirname(normalizePath(.file)))
fig_dir    <- file.path(repo_root, "figures")

set.seed(42)

# ---- IRT simulation setup -----------------------------------------------------
n_items <- 100
item_difficulties <- rnorm(n_items, 0, 1)   # difficulties ~ N(0, 1)
theta_true <- 0
n_samples <- 5000

irt_prob <- function(theta, difficulty, discrimination = 1) {
  1 / (1 + exp(-discrimination * (theta - difficulty)))
}

compute_accuracy <- function(y_pred, y_true) mean((y_pred > 0.5) == y_true)
compute_auc <- function(y_pred, y_true) {
  pos <- y_pred[y_true == 1]
  neg <- y_pred[y_true == 0]
  if (length(pos) == 0 || length(neg) == 0) return(0.5)
  mean(outer(pos, neg, ">")) + 0.5 * mean(outer(pos, neg, "=="))
}
compute_f1 <- function(y_pred, y_true) {
  yb <- as.numeric(y_pred > 0.5)
  tp <- sum(yb == 1 & y_true == 1)
  fp <- sum(yb == 1 & y_true == 0)
  fn <- sum(yb == 0 & y_true == 1)
  precision <- if (tp + fp > 0) tp / (tp + fp) else 0
  recall <- if (tp + fn > 0) tp / (tp + fn) else 0
  if (precision + recall > 0) 2 * precision * recall / (precision + recall) else 0
}
compute_rmse <- function(y_pred, y_true) sqrt(mean((y_pred - y_true)^2))

# Training data for the FIG baseline.
n_train <- 2000
train_data <- tibble(
  item_id = sample(1:n_items, n_train, replace = TRUE),
  student_id = sample(1:200, n_train, replace = TRUE)
) %>%
  mutate(
    theta = rnorm(n(), 0, 1.5),
    true_prob = irt_prob(theta, item_difficulties[item_id], 1),
    y = rbinom(n(), 1, true_prob)
  )

# One metric set for (theta_hat, a, sample_idx). The per-call set.seed(42+idx)
# makes every response draw deterministic, so fork-based mcmapply is exactly
# reproducible. Returns a named numeric vector.
compute_metrics_for_theta <- function(theta_hat, pred_discrimination, sample_idx) {
  true_probs <- irt_prob(theta_true, item_difficulties, discrimination = 1)
  set.seed(42 + sample_idx)
  y_true <- rbinom(n_items, 1, true_probs)
  pred_probs <- irt_prob(theta_hat, item_difficulties, discrimination = pred_discrimination)
  pred_probs <- pmax(0.001, pmin(0.999, pred_probs))  # clip for numerical stability
  fig_details <- tryCatch(
    fractional_information_gain_validation(
      y_pred_eval = pred_probs, y_eval = y_true,
      item_id_eval = 1:n_items, student_id_eval = rep(1, n_items),
      y_train = train_data$y, item_id_train = train_data$item_id,
      use_shrinkage = TRUE
    ),
    error = function(e) list(fig_v = NA_real_)
  )
  c(accuracy = compute_accuracy(pred_probs, y_true),
    AUC = compute_auc(pred_probs, y_true),
    F1 = compute_f1(pred_probs, y_true),
    RMSE = compute_rmse(pred_probs, y_true),
    FIG = fig_details$fig_v)
}

n_cores <- max(1L, detectCores() - 1L)

# Parallel sweep over a grid; aggregate (mean over MC draws) to long format.
# FIG column is relabelled to "FIG-V" only for display (matches the other plots).
aggregate_long <- function(grid, key) {
  res <- mcmapply(
    compute_metrics_for_theta,
    grid$theta_hat, grid$pred_discrimination, grid$sample_idx,
    mc.cores = n_cores
  )  # metrics (rows) x combinations (cols)
  bind_cols(dplyr::select(grid, all_of(key)), as_tibble(t(res))) %>%
    group_by(across(all_of(key))) %>%
    summarise(across(c(accuracy, AUC, F1, RMSE, FIG),
                     \(v) mean(v, na.rm = TRUE)), .groups = "drop") %>%
    pivot_longer(c(accuracy, AUC, F1, RMSE, FIG),
                 names_to = "metric", values_to = "value") %>%
    mutate(metric = factor(metric,
                           levels = c("accuracy", "AUC", "F1", "RMSE", "FIG"),
                           labels = c("accuracy", "AUC", "F1", "RMSE", "FIG-V")))
}

# Panel A: vary the ability estimate (theta_hat) at correct calibration a = 1.
cat("Panel A: sweeping ability error at a = 1 ...\n")
gridA <- expand_grid(theta_hat = seq(-2, 2, by = 0.25),
                     pred_discrimination = 1.0, sample_idx = 1:n_samples)
panelA <- aggregate_long(gridA, "theta_hat") %>%
  mutate(theta_error = theta_true - theta_hat)

# Panel B: vary the discrimination a (log-spaced 0.25..4) at correct ability.
cat("Panel B: sweeping calibration a at theta_hat = theta ...\n")
gridB <- expand_grid(theta_hat = theta_true,
                     pred_discrimination = 2^seq(-2, 2, length.out = 17),
                     sample_idx = 1:n_samples)
panelB <- aggregate_long(gridB, "pred_discrimination") %>%
  rename(a = pred_discrimination)

# ---- plot -------------------------------------------------------------------
# Small multiples: facet_grid row per metric, one column per plot. The metric
# names are left-side facet strips styled as y-axis titles; each metric row has
# its own y-scale, forced identical across the two columns via geom_blank.

# cowplot measures text on an internal null device that defaults to base pdf(),
# which cannot handle the UTF-8 labels; use a cairo null device instead.
set_null_device("cairo")
metric_colors <- c(
  "accuracy" = "#E74C3C",
  "AUC"      = "#F39C12",
  "F1"       = "#9B59B6",
  "RMSE"     = "#2C3E50",
  "FIG-V"    = "#27AE60"
)

# Unicode Greek (rendered via cairo) instead of plotmath: plotmath's hat() draws
# a wide accent, while the composed glyph gives a tight LaTeX-like theta-hat.
theta     <- "\u03b8"
theta_hat <- "\u03b8\u0302"   # theta + combining circumflex
nu        <- "\u03bd"

panelA <- panelA %>% mutate(strip = "Impact~of~bias~(italic(a) == 1)")
panelB <- panelB %>% mutate(strip = paste0("Miscalibration (", theta_hat, " = ", theta, ")"))

# Per-metric reference lines: chance level for accuracy/AUC, zero for FIG-V.
refs <- tibble(metric = factor(c("accuracy", "AUC", "FIG-V"),
                               levels = levels(panelA$metric)),
               yref = c(0.5, 0.5, 0))
refsA <- refs %>% mutate(strip = unique(panelA$strip))
refsB <- refs %>% mutate(strip = unique(panelB$strip))

# Per-metric y-ranges, shared across the two columns and widened to include each
# row's reference line; forced onto both plots through geom_blank below.
ranges <- bind_rows(panelA %>% select(metric, value),
                    panelB %>% select(metric, value)) %>%
  group_by(metric) %>%
  summarise(ymin = min(value), ymax = max(value), .groups = "drop") %>%
  left_join(refs, by = "metric") %>%
  mutate(
    ymin = pmin(ymin, coalesce(yref, ymin)),
    ymax = pmax(ymax, coalesce(yref, ymax)),
    pad  = 0.06 * (ymax - ymin),
    ymin = ymin - pad, ymax = ymax + pad
  ) %>%
  select(metric, ymin, ymax) %>%
  pivot_longer(c(ymin, ymax), values_to = "y")
blankA <- ranges %>% mutate(x = 0, strip = unique(panelA$strip))
blankB <- ranges %>% mutate(a = 1, strip = unique(panelB$strip))

col_theme <- function(p) {
  p +
    geom_line(linewidth = 0.9, alpha = 0.9) +
    geom_point(size = 1.1, alpha = 0.7) +
    scale_color_manual(values = metric_colors) +
    scale_y_continuous(n.breaks = 4) +
    theme_cowplot(font_size = 12, font_family = "DejaVu Sans") +
    panel_border() +
    theme(legend.position = "none",
          axis.text = element_text(size = 8.5),
          axis.title.x = element_text(size = 13),
          strip.text.x = element_text(size = 11))
}

pA <- col_theme(
  ggplot(panelA, aes(theta_error, value, color = metric)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray50", alpha = 0.7) +
    geom_hline(data = refsA, aes(yintercept = yref),
               linetype = "dotted", color = "gray50", alpha = 0.8) +
    geom_blank(data = blankA, aes(x = x, y = y), inherit.aes = FALSE)
) +
  facet_grid(metric ~ strip, labeller = labeller(strip = label_parsed),
             scales = "free_y", switch = "y") +
  scale_x_continuous(breaks = seq(-2, 2, 1)) +
  labs(x = paste0(nu, " = ", theta, " \u2212 ", theta_hat), y = NULL) +
  theme(strip.placement = "outside",
        strip.background.y = element_blank(),
        strip.text.y.left = element_text(angle = 90, size = 11))

pB <- col_theme(
  ggplot(panelB, aes(a, value, color = metric)) +
    geom_vline(xintercept = 1, linetype = "dashed", color = "gray50", alpha = 0.7) +
    geom_hline(data = refsB, aes(yintercept = yref),
               linetype = "dotted", color = "gray50", alpha = 0.8) +
    geom_blank(data = blankB, aes(x = a, y = y), inherit.aes = FALSE)
) +
  facet_grid(metric ~ strip, scales = "free_y") +
  scale_x_continuous(transform = "log2", breaks = c(0.25, 0.5, 1, 2, 4),
                     labels = c("0.25", "0.5", "1", "2", "4")) +
  labs(x = expression(italic(a)), y = NULL) +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank(),
        strip.text.y = element_blank(), strip.background.y = element_blank())

row <- plot_grid(pA, pB, nrow = 1, align = "h", axis = "tb", rel_widths = c(1.18, 1))
title <- ggdraw() +
  draw_label("Metric sensitivity to bias and miscalibration",
             fontface = "bold", size = 14, fontfamily = "DejaVu Sans", x = 0.06, hjust = 0)
combined <- plot_grid(title, row, ncol = 1, rel_heights = c(0.07, 1))

out_path <- file.path(fig_dir, "10b_track_metrics_col.pdf")
ggsave(out_path, combined, width = 6, height = 5.0, dpi = 300, bg = "white", device = cairo_pdf)
cat("Plot saved to:", out_path, "\n")
