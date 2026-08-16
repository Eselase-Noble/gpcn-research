# Experiments to run — fills every `[TO RUN]` in paper/main.tex

Each `\todo{...}` (red) marker in the manuscript maps to one run below. Do them in
priority order; 1–2 are the ones that decide publishability.

## P1 — Naive-fusion baselines (defends the fusion *mechanism*) ⭐ highest value
Isolates whether cross-magnification attention beats dumb fusion of the same four streams.

```bash
# (a) FREE — no retrain — post-hoc late fusion from the already-trained model:
python -c "from config import get_full_config; from fusion_baselines import run_posthoc_fusion_baselines; \
cfg=get_full_config(); cfg.experiment_name='gpcn_multimag'; run_posthoc_fusion_baselines(cfg)"

# (b) CLEAN — retrain the two from-scratch variants under the identical recipe:
#     set cfg.model.fusion_mode and run the normal multimag training each time.
for MODE in mean concat none; do
  python -c "from config import get_full_config; from multimag import train_multimag; \
from fusion_baselines import build_fusion_model; import multimag; \
multimag.build_multimag_model=build_fusion_model; \
cfg=get_full_config(); cfg.model.fusion_mode='$MODE'; cfg.experiment_name='fusion_$MODE'; train_multimag(cfg)"
done
```
→ Fills **Table `tab:fusion-baselines`** (accuracy/MCC/AUC + McNemar p per row).
Note: `build_fusion_model` is monkey-patched over `build_multimag_model` so the existing
train loop trains each arm unchanged. If you prefer, edit `multimag.py` to import it directly.

## P2 — GPCN on/off ablation (defends contribution #1: the graph) ⭐
Already wired in `experiments.py`. Run both arms across 3 seeds with the built-in paired test:
```bash
python experiments.py --mode ablation --mag 40X --seeds 0 1 2
# writes results/ablation_40X_significance.json  (mean±sd, paired t-test)
```
→ Fills **Table `tab:gpcn`**. Optionally repeat per magnification.

## P3 — McNemar: fused vs best single stream
Add to the single-mag ablation run (paired predictions already available):
```python
from fusion_baselines import mcnemar_test   # feed y_true, best-single preds, fused preds
```
→ Fills the `\todo{$p$ vs.\ best single stream}` in §ablation.

## P4 — Learned magnification weights (interpretability number)
Printed automatically by `single_mag_baseline.run_single_mag_baseline` (softmax(mag_weights)).
→ Fills the `\todo{report the four weights...}` in §"Learned magnification weights".

## P5 — (strong-venue only) k-fold patient-level CV + external validation
- `kfold_cv.py` already present — run 5-fold patient-level CV for the fusion model → error bars.
- BACH (`bach.py`, `BACH_Download_Setup.ipynb`) for cross-cohort external validation.
Only needed if targeting Medical Image Analysis / IEEE TMI / MICCAI main.

---
### After each run
Replace the corresponding `\todo{...}` in `paper/main.tex` with the number, and delete the
`\newcommand{\todo}` line once all are filled (compilation will then flag any you missed).
