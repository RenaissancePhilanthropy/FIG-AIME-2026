#' Item-Only baseline vs Trained SAKT across datasets
#'
#' Three panels (Accuracy, AUC, FIG-V) with dataset on the x-axis and dodged
#' light/dark bars for the Item-Only model vs the Trained SAKT model, both
#' evaluated on the same by-student splits. Visual style mirrors
#' fig1_item_info_bias.R (theme_cowplot, faceted metrics, dotted baseline lines).
#'
#' Reads pipeline/out/baseline_vs_sota_computed.csv (written by compute_table.py); the FIG
#' panel shows the per-student FIG-V (each student weighted equally).
#'
#' Paper Figure 2.

library(tidyverse)
library(cowplot)

# Resolve the repo root from this script's path (run via Rscript, e.g. through the Makefile).
.file <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
repo_root <- dirname(dirname(normalizePath(.file)))
csv_path <- file.path(repo_root, "pipeline", "out", "baseline_vs_sota_computed.csv")

out_path <- file.path(repo_root, "figures", "item_only_vs_trained_perstudent.pdf")

# --- tidy: dataset x model x metric -> value ---------------------------------
dataset_levels <- c("as15", "al2005", "bd2006", "nips34", "sim_u2", "sim_u5")
dataset_labels <- c(
  sim_u2 = "Sim U[-2,2]", sim_u5 = "Sim U[-5,5]",
  as15 = "AS'15", al2005 = "AL'05", bd2006 = "BD'06", nips34 = "NIPS34"
)

raw <- read_csv(csv_path, show_col_types = FALSE)

model_levels <- c("Item-only", "Trained SAKT", "Reference")

plot_data <- raw %>%
  transmute(
    key,
    `Item-only_Accuracy`    = accuracy,
    `Item-only_AUC`         = auc,
    `Item-only_FIG`         = fig_student,
    `Trained SAKT_Accuracy` = t_acc,
    `Trained SAKT_AUC`      = t_auc,
    `Trained SAKT_FIG`      = t_figs,
    `Reference_Accuracy`    = best_acc,
    `Reference_AUC`         = best_auc,
    # Bayes-optimal FIG (true-probability predictor) for the sim sets; NA for the
    # real datasets, where no FIG ceiling is known -> they keep two FIG bars.
    `Reference_FIG`         = best_figs
  ) %>%
  pivot_longer(
    -key,
    names_to = c("model", "metric"),
    names_sep = "_",
    values_to = "value"
  ) %>%
  filter(!is.na(value)) %>%             # drops Reference FIG for the real datasets
  mutate(
    dataset = factor(dataset_labels[key], levels = dataset_labels[dataset_levels]),
    metric  = factor(metric, levels = c("Accuracy", "AUC", "FIG")),
    model   = factor(model, levels = model_levels)
  ) %>%
  # centre each dataset's present bars: 3 where a reference exists (all of
  # Accuracy/AUC; only the sim sets on FIG), 2 otherwise (real datasets on FIG).
  group_by(metric, dataset) %>%
  arrange(model, .by_group = TRUE) %>%
  mutate(n_g = n(), xc = as.integer(dataset) + (row_number() - (n_g + 1) / 2) * 0.27) %>%
  ungroup()

# 95% student-bootstrap CIs (Item-only & Trained SAKT; Reference has none).
ci_long <- raw %>%
  transmute(
    key,
    `Item-only_Accuracy_lo` = acc_lo,    `Item-only_Accuracy_hi` = acc_hi,
    `Item-only_AUC_lo`      = auc_lo,    `Item-only_AUC_hi`      = auc_hi,
    `Item-only_FIG_lo`      = figs_lo,   `Item-only_FIG_hi`      = figs_hi,
    `Trained SAKT_Accuracy_lo` = t_acc_lo,  `Trained SAKT_Accuracy_hi` = t_acc_hi,
    `Trained SAKT_AUC_lo`      = t_auc_lo,  `Trained SAKT_AUC_hi`      = t_auc_hi,
    `Trained SAKT_FIG_lo`      = t_figs_lo,  `Trained SAKT_FIG_hi`      = t_figs_hi
  ) %>%
  pivot_longer(-key, names_to = c("model", "metric", "bound"), names_sep = "_") %>%
  pivot_wider(names_from = bound, values_from = value)

plot_data <- plot_data %>%
  mutate(model = as.character(model), metric = as.character(metric)) %>%
  left_join(ci_long, by = c("key", "model", "metric")) %>%
  mutate(
    metric = factor(metric, levels = c("Accuracy", "AUC", "FIG")),
    model  = factor(model, levels = model_levels),
    # Colour key: the sim Reference bars are the Bayes-optimal ceiling, drawn
    # black to set them apart from the published-model Reference (real datasets).
    # Bar POSITION is unaffected -- it is still set by `model` (above), so these
    # stay in the third slot of each sim cluster.
    fill_key = factor(
      ifelse(as.character(model) == "Reference" & grepl("^sim_", key),
             "Bayes-optimal", as.character(model)),
      levels = c(model_levels, "Bayes-optimal")
    )
  )

# One label per dataset cluster on the Accuracy/AUC panels: the item-only metric value
# (2 decimal places), placed at the top of the cluster (its tallest bar).
annot <- plot_data %>%
  group_by(metric, dataset) %>%
  summarise(
    x = as.integer(dataset)[1],
    y = max(value),
    item_val = value[model == "Item-only"][1],
    .groups = "drop"
  ) %>%
  mutate(label = sprintf("%.2f", item_val)) %>%
  filter(metric != "FIG")   # no in-figure numbers on the FIG panel

# --- plot --------------------------------------------------------------------
# Each metric is its own panel so it can have its own baseline and y-limits:
# bars rise from the panel's baseline (Accuracy 0.5, AUC 0.5 = chance, FIG 0),
# with a dotted line at that baseline and a small gap below it so all panels
# share the same bottom. A third bar per dataset (Reference) shows the best
# published item-ID-only model, or the Bayes-optimal true-probability predictor
# for the Sim sets. On the FIG panel only the sim sets have a (Bayes-optimal)
# reference; the real datasets have no known FIG ceiling, so they keep two bars.
# Panels are stitched with cowplot::plot_grid.
fill_vals <- c(            # light -> medium -> dark (more information = darker)
  "Item-only"     = "#C6DBEF",
  "Trained SAKT"  = "#6BAED6",
  "Reference"     = "#08519C",
  "Bayes-optimal" = "#000000"   # the sim ceiling, drawn black to stand apart
)
bar_hw <- 0.12                          # bar half-width in x units
x_breaks <- seq_along(levels(plot_data$dataset))
x_labels <- levels(plot_data$dataset)

panel <- function(metric_name, base, ylim, breaks = waiver(), strip_label = metric_name) {
  ggplot(filter(plot_data, metric == metric_name)) +
    geom_rect(
      aes(xmin = xc - bar_hw, xmax = xc + bar_hw, ymin = base, ymax = value, fill = fill_key)
    ) +
    geom_hline(yintercept = base, linetype = "dotted", color = "gray40") +
    geom_errorbar(
      data = filter(plot_data, metric == metric_name, !is.na(lo)),
      aes(x = xc, ymin = lo, ymax = hi),
      width = 0.07, linewidth = 0.3, colour = "gray25", inherit.aes = FALSE
    ) +
    geom_text(
      data = filter(annot, metric == metric_name),
      aes(x = x, y = y, label = label),
      vjust = -0.5, size = 3, colour = "gray20", inherit.aes = FALSE
    ) +
    # single-level facet -> strip header (relabelled so the FIG panel states variant)
    facet_wrap(~metric, labeller = as_labeller(setNames(strip_label, metric_name))) +
    scale_fill_manual(values = fill_vals, drop = FALSE) +
    scale_x_continuous(breaks = x_breaks, labels = x_labels) +
    scale_y_continuous(breaks = breaks) +
    coord_cartesian(xlim = c(0.4, max(x_breaks) + 0.6), ylim = ylim, expand = FALSE) +
    labs(x = NULL, y = NULL, fill = NULL) +
    theme_cowplot(font_size = 12) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
      legend.position = "none",
      plot.margin = margin(5, 8, 5, 5)
    )
}

# A small gap below each baseline keeps the bottoms even. Accuracy & AUC share 0.1
# tick spacing; FIG uses 0.02 ticks.
p_acc <- panel("Accuracy", base = 0.5,   ylim = c(0.485, 1.0),   breaks = seq(0.5, 1, 0.1))
p_auc <- panel("AUC",      base = 0.5, ylim = c(0.485, 1.0),   breaks = seq(0.5, 1, 0.1))
# The FIG panel dips below 0 (item base rates slightly mispredict the typical student), so the
# axis extends negative; the top leaves room for the ~0.13 Bayes-optimal sim bars.
p_fig <- panel("FIG", base = 0, ylim = c(-0.10, 0.15), breaks = seq(-0.08, 0.14, 0.02),
               strip_label = "FIG-V (per-student)")

legend <- get_legend(
  p_acc + theme(legend.position = "top", legend.justification = "center")
)
row <- plot_grid(p_acc, p_auc, p_fig, nrow = 1)
title <- ggdraw() +
  draw_label("Prediction metrics for widely-used datasets",
             fontface = "bold", size = 13, x = 0.01, hjust = 0)
subtitle <- ggdraw() +
  draw_label("Number = prediction using item base rates.",
             size = 9.5, colour = "gray30", x = 0.01, hjust = 0)
p <- plot_grid(title, subtitle, legend, row, ncol = 1,
               rel_heights = c(0.09, 0.07, 0.1, 1))

ggsave(out_path, p, width = 9, height = 3.6, dpi = 300, bg = "white")
cat("Plot saved to:", out_path, "\n")
