#' Experiment 11: FIG increases with observations (per-student, single panel)
#'
#' One panel, three series of per-student FIG vs. the number of Skill-3 responses
#' observed (Skill-3-input protocol). All three series come from
#' pipeline/out/calibration_curves.csv:
#'   - All items   : FIG on ALL held-out items as Skill-3 items are revealed
#'   - Skill 3     : FIG on held-out Skill-3 items as Skill-3 items are revealed
#'   - Other skills: FIG on held-out other-skill items, same Skill-3-reveal run
#'
#' Skill-3 FIG rises as more Skill-3 context is shown; Other-skills and All-items
#' FIG stay near zero (skills are independent by construction, so only the
#' observed skill's FIG benefits).
#'
#' Paper Figure 4.

library(tidyverse)
library(cowplot)

# Resolve the repo root from this script's path (run via Rscript, e.g. through the Makefile).
.file <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
repo_root <- dirname(dirname(normalizePath(.file)))
fig_dir    <- file.path(repo_root, "figures")

# Per-student FIG-by-observation-count, written by the pipeline (make train).
temp <- read_csv(
  file.path(repo_root, "pipeline", "out", "calibration_curves.csv"),
  show_col_types = FALSE
)

# Shared aesthetic: color = item group.
group_colors <- c("All items" = "#984EA3", "Skill 3" = "#2C7BB6", "Other skills" = "#E31A1C")

# Three per-student FIG series with 95% student-bootstrap CIs, all from Skill-3-input protocol.
# Cap at 16 revealed Skill-3 responses: beyond this the held-out Skill-3 set gets too small
# (only 8 of 40 remain at ct=32 -> noisy, and All-items dips). At ct=16, 24 of 40 are still held out.
skill3_rows <- temp %>% filter(ct <= 16)

plot_df <- bind_rows(
  skill3_rows %>% transmute(ct, group = "All items",    value = all_fig_v,    lo = all_fig_v_lo,    hi = all_fig_v_hi),
  skill3_rows %>% transmute(ct, group = "Skill 3",      value = skill3_fig_v, lo = skill3_fig_v_lo, hi = skill3_fig_v_hi),
  skill3_rows %>% transmute(ct, group = "Other skills", value = other_fig_v,  lo = other_fig_v_lo,  hi = other_fig_v_hi)
) %>%
  mutate(group = factor(group, levels = c("All items", "Skill 3", "Other skills")))

# Use data-driven y-limits that include the full CI range.
y_max <- max(plot_df$hi, na.rm = TRUE) * 1.1
y_min <- min(min(plot_df$lo, na.rm = TRUE) - 0.005, -0.015)

p <- ggplot(plot_df, aes(x = ct, y = value, color = group)) +
  geom_hline(yintercept = 0, linetype = "dotted") +
  geom_line(aes(group = group), linewidth = 1) +
  geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.6, linewidth = 0.6) +
  geom_point(size = 2) +
  scale_color_manual(values = group_colors) +
  scale_y_continuous(limits = c(y_min, y_max), breaks = seq(0, 0.10, 0.02)) +
  scale_x_continuous(limits = c(0, 16.5), breaks = c(0, 4, 8, 12, 16)) +
  labs(
    title = "FIG-V increases with observations",
    subtitle = "Only the observed skill's FIG-V rises",
    x = "Skill 3 responses observed",
    y = "FIG-V (per-student)",
    color = NULL
  ) +
  theme_cowplot(font_size = 14) +
  theme(legend.position = "bottom",
        legend.justification = "center",
        axis.text.x = element_text(size = 12),
        axis.title.x = element_text(size = 15),
        plot.subtitle = element_text(size = 12, colour = "gray30")) +
  panel_border()

out_path <- file.path(fig_dir, "11_calibration_fig_combined.pdf")
ggsave(out_path, p, width = 6, height = 4.2, dpi = 300, bg = "white")
cat("Plot saved to:", out_path, "\n")
