# GPCN-ViT — Ablation & Validation Run Plan

This is the ordered, crash-safe schedule to take the project from "good single-split
numbers" to "publishable, defensible results" for **BreakHis** and **BACH**.

Everything here is **resumable on Colab**: per-epoch checkpoints (`checkpoint_*_epoch_N.pth`),
per-fold caches (`cv_*_fold*.json`), and per-run caches (`*_seed*.json`) mean a disconnect
costs at most the in-progress epoch/fold/run. Re-running the same command continues where it stopped.

> Defaults: backbone = Phikon (`hf-hub:1aurent/vit_base_patch16_224.owkin_pancancer`),
> `num_epochs=50`, `batch_size=16`. Override epochs with `--epochs` for quick smoke tests.

---

## Priority order (do them in this sequence)

The novelty of this paper is the **GPCN module**, not "ViT on histopathology" (that space is
crowded — fine-tuned ViT already hits ~99.99% on BreakHis binary; DWNAT-Net ~91.25% on BACH).
So the experiments are ordered by **how much they protect the central claim**.

| # | Experiment | Why it matters | Blocks publication? |
|---|---|---|---|
| 1 | **BACH ablation** (GPCN on/off × 3 seeds) | Proves GPCN helps on the 4-class task, *with a p-value* | **Yes** — core claim |
| 2 | **BreakHis ablation** @ 200X (GPCN on/off × 3 seeds) | Same proof on the binary task | **Yes** — core claim |
| 3 | **BACH 5-fold CV** | Replaces the n=80 single split with mean±std + 95% CI | **Yes** — n=80 is too small alone |
| 4 | **BreakHis 5-fold CV** @ 200X (patient-level) | Shows the headline number isn't a lucky split | **Yes** |
| 5 | Calibration report (both) | ECE/MCE before vs after temperature scaling | Strongly recommended |
| 6 | BreakHis per-magnification + SOTA table | Completes the standard BreakHis protocol | Recommended |

Items 1–4 are the difference between "rejected: insufficient validation" and "accepted".
Calibration (5) now runs **automatically** at the end of every BACH training run.

---

## Pacing (suggested calendar)

Each "run" = one full train (~50 epochs). Treat a Colab session as ~one batch of runs;
checkpoints make multi-session safe.

- **Session 1 — BACH ablation (6 runs).** Highest value, smallest dataset (400 images) so
  it finishes fastest. This single command produces the headline "GPCN contribution" table.
- **Session 2 — BACH 5-fold CV (5 runs).** Stable BACH numbers with confidence intervals.
- **Session 3–4 — BreakHis ablation @ 200X (6 runs).** Larger dataset → budget more time.
- **Session 5–6 — BreakHis 5-fold CV @ 200X (5 runs).**
- **Session 7 (optional) — BreakHis per-magnification (4 runs) + SOTA table.**

If GPU time is tight, drop seeds from 3 → 2 (still gives a paired test) before dropping any
experiment. Never report the ablation from a single seed.

---

## Commands

### 1. BACH ablation — GPCN on/off, paired over seeds (CORE)
```bash
python bach.py --bach_root "$BACH_PATH" --mode ablation --seeds 0 1 2 --results_dir results
```
Produces `results/bach_ablation_summary.json` with a **paired t-test per metric**
(accuracy, balanced_accuracy, f1_macro, auc_macro_ovr, mcc, ece) → mean±std, Δ, 95% CI, p-value.

### 2. BACH 5-fold cross-validation
```bash
python bach.py --bach_root "$BACH_PATH" --mode kfold --n_folds 5 --results_dir results
```
Produces `cv_bach_summary.{json,csv}` (mean±std + 95% CI across folds, image-level stratified).

### 3. BreakHis ablation @ 200X — GPCN on/off, paired over seeds (CORE)
Notebook (recommended, lets you pass dataset paths):
```python
from experiments import run_ablation
run_ablation(mag='200X', seeds=(0, 1, 2), epochs=50,
             data_root=BREAKHIS_PATH, save_dir=OUTPUT_PATH, results_dir='results')
```
Produces `results/ablation_200X_significance.json`.

### 4. BreakHis 5-fold CV @ 200X (patient-level, leakage-free)
```bash
python kfold_cv.py --data_root "$BREAKHIS_PATH" --save_dir "$OUTPUT_PATH" --mag 200X --n_folds 5 --epochs 50
```

### 5. Calibration (BACH = automatic; BreakHis = on demand)
BACH training already emits `bach_reliability.png` + calibrated ECE/MCE. For a BreakHis
checkpoint:
```python
from calibration_report import report_from_checkpoint
from config import get_full_config
cfg = get_full_config(); cfg.data.data_root = BREAKHIS_PATH; cfg.data.train_magnification = '200X'
report_from_checkpoint(cfg, 'path/to/best_model.pth', save_path='breakhis_reliability.png', show=False)
```

### 6. BreakHis per-magnification + SOTA comparison table
```python
from experiments import run_all_magnifications, build_results_table, build_comparison_table
recs = run_all_magnifications(data_root=BREAKHIS_PATH, save_dir=OUTPUT_PATH, epochs=50)
print(build_results_table(recs)); print(build_comparison_table(recs))
```

---

## What goes in the paper

- **Ablation tables** (BACH + BreakHis): GPCN ON vs OFF, mean±std over seeds, Δ with 95% CI
  and paired t-test p-value. This is the evidence that GPCN — not just the ViT backbone — drives performance.
- **CV tables** (BACH + BreakHis): every metric as mean ± std and 95% CI. Report these as the
  headline numbers, not the single split.
- **Reliability diagrams** + ECE/MCE before/after temperature scaling.
- **SOTA comparison table** — ⚠️ the numbers pre-filled in `experiments.PUBLISHED_SOTA` are
  tagged `[VERIFY]`; confirm each against its source (protocols differ: image- vs patient-level,
  binary vs multiclass) before submission.

## Honest gaps to still close before submitting
1. **Run the experiments above** — the code is ready; the numbers are not yet generated.
2. **Verify SOTA citations** in `experiments.py`.
3. Decide the BACH protocol framing: this uses **image-level** stratified splits (BACH Part A
   has no patient mapping) — state it explicitly so reviewers don't expect patient-level.
