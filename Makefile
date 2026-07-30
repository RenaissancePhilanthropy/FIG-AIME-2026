# FIG-AIME-2026 -- reproduce the paper's four figures.
#
# Figures 1 & 3 are self-contained R scripts; Figures 2 & 4 need their input data
# produced first (data + train), then plotted (plots). See README.md.
#
# make does NOT install anything -- install the R/python dependencies first
# (install commands are in README.md), then run these targets.
#
# Override these on the command line as needed:
#   RSCRIPT  Rscript executable            (default: Rscript)
#   PY       python from the pipeline venv (default: python)
# The `data`/`train` Python env is declared in pyproject.toml; create it with `uv sync`
# and run those targets via `uv run make data` (PY then resolves to the uv env). See README.

RSCRIPT ?= Rscript
PY      ?= python
# Extra flags forwarded to the SAKT trainers (both fig4_calibration_curves.py and
# train_sakt_long.py accept --max-epochs/--patience), e.g. for a quick smoke run:
#   make train SAKT_FLAGS="--max-epochs 3"
SAKT_FLAGS    ?=

# --- paths -------------------------------------------------------------------
R_DIR    := R
PIPE     := pipeline
OUT      := $(PIPE)/out
DATADIR  := $(PIPE)/datasets

DATA_1PL    := $(DATADIR)/1pl/training_1pl_5000_students_200_items.csv
SIM2_TR     := $(DATADIR)/sim_irt/sim_irt_unif2/sim_irt_unif2_train.csv
SIM2_TE     := $(DATADIR)/sim_irt/sim_irt_unif2/sim_irt_unif2_test.csv
SIM5_TR     := $(DATADIR)/sim_irt/sim_irt_unif5/sim_irt_unif5_train.csv
SIM5_TE     := $(DATADIR)/sim_irt/sim_irt_unif5/sim_irt_unif5_test.csv
AS15_TR     := $(DATADIR)/assistments2015/assistments2015_train.csv
AS15_TE     := $(DATADIR)/assistments2015/assistments2015_test.csv
NIPS34_TR   := $(DATADIR)/nips34/nips34_train.csv
NIPS34_TE   := $(DATADIR)/nips34/nips34_test.csv
AL2005_TR   := $(DATADIR)/kddcup_al2005/al2005_train.csv
AL2005_TE   := $(DATADIR)/kddcup_al2005/al2005_test.csv
BD2006_TR   := $(DATADIR)/kddcup_bd2006/bd2006_train.csv
BD2006_TE   := $(DATADIR)/kddcup_bd2006/bd2006_test.csv

# Gated raw files, fetched by hand from their distributors (see pipeline/DATASETS.md). Copy them
# to these canonical paths to have `make data` auto-discover them; override to use other paths.
AS15_RAW    ?= $(DATADIR)/assistments2015/2015_100_skill_builders_main_problems.csv
AL2005_RAW  ?= $(DATADIR)/kddcup_al2005/algebra_2005_2006_train.txt
BD2006_RAW  ?= $(DATADIR)/kddcup_bd2006/bridge_to_algebra_2006_2007_train.txt

OUT_CALIB := $(OUT)/calibration_curves.csv
OUT_TABLE := $(OUT)/baseline_vs_sota_computed.csv

# key:train:test triples consumed by the SAKT training loop in `train`.
SAKT_SPECS := \
  sim_u2:$(SIM2_TR):$(SIM2_TE) \
  sim_u5:$(SIM5_TR):$(SIM5_TE) \
  as15:$(AS15_TR):$(AS15_TE) \
  nips34:$(NIPS34_TR):$(NIPS34_TE) \
  al2005:$(AL2005_TR):$(AL2005_TE) \
  bd2006:$(BD2006_TR):$(BD2006_TE)

.DEFAULT_GOAL := help
.PHONY: help check plots data data-kddcup train clean

help:
	@echo "FIG-AIME-2026 -- reproduce the paper figures"
	@echo
	@echo "Figures 1 & 3 are self-contained; Figures 2 & 4 need the pipeline run first."
	@echo "Compare against figures/reference/."
	@echo
	@echo "  make plots               draw the figures -> figures/  (Figs 2 & 4 are skipped"
	@echo "                           until you have run 'make data' / 'make train')"
	@echo
	@echo "Produce the Fig 2 & 4 inputs (needs the python venv):"
	@echo "  make data                generate synthetic data + prepare the real datasets"
	@echo "                           + item-only table  (no SAKT training)"
	@echo "  make data-kddcup AL2005_RAW=... BD2006_RAW=...   add the gated KDD Cup columns"
	@echo "  make train               train SAKT (Fig 4 curve + Fig 2 bars)"
	@echo
	@echo "Helpers:"
	@echo "  make check               report whether the R / python deps are present"
	@echo "  make clean               remove generated figures and pipeline outputs"
	@echo
	@echo "Dependencies are installed manually (see README); make never installs them."
	@echo "Vars: RSCRIPT=$(RSCRIPT)  PY=$(PY)"

# --- prerequisite check ------------------------------------------------------
check:
	@echo "== R =="
	@command -v $(RSCRIPT) >/dev/null 2>&1 && echo "  Rscript: OK ($$($(RSCRIPT) --version 2>&1 | head -1))" || echo "  Rscript: MISSING"
	@$(RSCRIPT) -e 'q(status=!requireNamespace("figmetric",quietly=TRUE))' >/dev/null 2>&1 && echo "  figmetric: OK" || echo "  figmetric: MISSING (install commands in README: Draw the figures)"
	@$(RSCRIPT) -e 'q(status=!all(sapply(c("tidyverse","cowplot","pROC"),requireNamespace,quietly=TRUE)))' >/dev/null 2>&1 && echo "  tidyverse/cowplot/pROC: OK" || echo "  tidyverse/cowplot/pROC: MISSING (install.packages(c('tidyverse','cowplot','pROC')))"
	@echo "== python (PY=$(PY)) =="
	@command -v $(PY) >/dev/null 2>&1 && echo "  python: OK" || echo "  python: MISSING"
	@$(PY) -c "import pandas,sklearn,fig" >/dev/null 2>&1 && echo "  pandas/sklearn/fig: OK" || echo "  pandas/sklearn/fig: MISSING (see pipeline/README.md)"
	@$(PY) -c "import torch" >/dev/null 2>&1 && echo "  torch (needed for 'make train'): OK" || echo "  torch: MISSING (needed for 'make train'; see pipeline/README.md)"

# --- draw the figures --------------------------------------------------------
# Figs 1 & 3 are self-contained. Figs 2 & 4 read pipeline/out/ and render only if you
# have run 'make data' / 'make train'; otherwise they are skipped with a note.
plots:
	$(RSCRIPT) $(R_DIR)/fig1_item_info_bias.R
	$(RSCRIPT) $(R_DIR)/fig3_metric_sensitivity.R
	@if [ -f $(OUT_TABLE) ]; then $(RSCRIPT) $(R_DIR)/fig2_item_only_vs_trained.R; else echo "[fig2] SKIP: run 'make data' (and 'make train') first -- $(OUT_TABLE) missing"; fi
	@if [ -f $(OUT_CALIB) ]; then $(RSCRIPT) $(R_DIR)/fig4_fig_vs_observations.R; else echo "[fig4] SKIP: run 'make train' first -- $(OUT_CALIB) missing"; fi
	@echo "Wrote available figures to figures/ (compare against figures/reference/)."

# --- produce the figure inputs from data --------------------------------------
# Generate synthetic datasets, download+prepare the open real datasets, and compute the
# item-only baseline table. SAKT training is the separate `train`.
data:
	@mkdir -p $(OUT)
	@test -f $(DATA_1PL) || $(PY) $(PIPE)/synthetic/generate_1pl_5skill.py
	@test -f $(SIM2_TR)  || $(PY) $(PIPE)/synthetic/generate_sim_irt.py
	@if [ ! -f $(AS15_TR) ] && [ -f "$(AS15_RAW)" ]; then $(PY) $(PIPE)/datasets/assistments2015/prepare_assistments2015.py --raw "$(AS15_RAW)"; fi
	@test -f $(NIPS34_TR)|| $(PY) $(PIPE)/datasets/nips34/prepare_nips34.py
	@if [ ! -f $(AL2005_TR) ] && [ -f "$(AL2005_RAW)" ]; then $(PY) $(PIPE)/datasets/prepare_kddcup.py --dataset al2005 --raw "$(AL2005_RAW)"; fi
	@if [ ! -f $(BD2006_TR) ] && [ -f "$(BD2006_RAW)" ]; then $(PY) $(PIPE)/datasets/prepare_kddcup.py --dataset bd2006 --raw "$(BD2006_RAW)"; fi
	$(PY) $(PIPE)/compute_table.py
	@if [ ! -f $(AS15_TR) ]; then \
	  echo; \
	  echo "NOTE: the as15 column is gated -- not auto-downloaded. To include it, get the 2015"; \
	  echo "      skill builder CSV from ASSISTments (see pipeline/DATASETS.md) and copy it to:"; \
	  echo "        $(AS15_RAW)"; \
	  echo "      then re-run 'make data' -- it is auto-discovered at that path."; \
	fi
	@if [ ! -f $(AL2005_TR) ] || [ ! -f $(BD2006_TR) ]; then \
	  echo; \
	  echo "NOTE: the KDD Cup columns (al2005, bd2006) are gated -- not auto-downloaded. To"; \
	  echo "      include them, copy the raw files (from PSLC DataShop; see pipeline/DATASETS.md) to:"; \
	  echo "        $(AL2005_RAW)"; \
	  echo "        $(BD2006_RAW)"; \
	  echo "      then re-run 'make data' -- they are auto-discovered at those paths."; \
	fi
	@echo "Next: 'make train' (SAKT bars + Fig 4 curve), then 'make plots'."

# Gated KDD Cup datasets: explicitly (re)prepare from the raw files. Reads them from the
# canonical AL2005_RAW/BD2006_RAW paths (override for a custom location); errors if not found.
# `make data` also auto-prepares these when the raw files are present, so this is optional.
data-kddcup:
	@test -f "$(AL2005_RAW)" || { echo "ERROR: raw file not found: $(AL2005_RAW)"; echo "       Download from PSLC DataShop and copy it there, or pass AL2005_RAW=/your/path (see pipeline/DATASETS.md)."; exit 1; }
	@test -f "$(BD2006_RAW)" || { echo "ERROR: raw file not found: $(BD2006_RAW)"; echo "       Download from PSLC DataShop and copy it there, or pass BD2006_RAW=/your/path (see pipeline/DATASETS.md)."; exit 1; }
	$(PY) $(PIPE)/datasets/prepare_kddcup.py --dataset al2005 --raw "$(AL2005_RAW)"
	$(PY) $(PIPE)/datasets/prepare_kddcup.py --dataset bd2006 --raw "$(BD2006_RAW)"
	$(PY) $(PIPE)/compute_table.py
	@echo "Prepared al2005, bd2006. Run 'make train' to add their SAKT bars."

# Train SAKT for Fig 4 (synthetic) and the Fig 2 bars (each prepared dataset), then
# re-assemble the table. Per-dataset training is skipped if already done
# (rm pipeline/out/.sakt_<key> to force a retrain).
train:
	@mkdir -p $(OUT)
	@test -f $(DATA_1PL) || { echo "ERROR: run 'make data' first (synthetic data missing)."; exit 1; }
	@test -f $(OUT_CALIB) || $(PY) $(PIPE)/fig4_calibration_curves.py $(SAKT_FLAGS)
	@for spec in $(SAKT_SPECS); do \
	  key=$${spec%%:*}; rest=$${spec#*:}; tr=$${rest%%:*}; te=$${rest##*:}; \
	  if [ ! -f "$$tr" ]; then echo "[skip] $$key: not prepared"; continue; fi; \
	  if [ -f "$(OUT)/.sakt_$$key" ]; then echo "[skip] $$key: already trained (rm $(OUT)/.sakt_$$key to retrain)"; continue; fi; \
	  echo "== training SAKT: $$key =="; \
	  $(PY) $(PIPE)/train_sakt_long.py --train "$$tr" --test "$$te" --dataset "$$key" $(SAKT_FLAGS) && touch "$(OUT)/.sakt_$$key"; \
	done
	$(PY) $(PIPE)/compute_table.py
	@echo "Trained SAKT + assembled $(OUT_TABLE). Run 'make plots'."

clean:
	rm -f figures/*.pdf figures/*.png
	rm -rf $(OUT)
	@echo "Removed generated figures and $(OUT). Downloaded/generated datasets kept."
