# VeriFrame v1+ (default dashboard model)

**Data:** only the original v1 videos (1,299 FakeTT videos, `data/features_v1_1299.npz`) and the original
temporal split (train 962 / val 171 / test 166). No extra videos, no audio, no speech, no publisher bio.

**What changed vs v1**
- Richer frame pooling: mean + std + max + mean absolute frame-to-frame change of CLIP embeddings.
- Regularisation (LogReg C, SVM C) chosen on the validation set.
- New member: block-wise late fusion (one LogReg per modality block -> meta LogReg on out-of-fold scores).
- Fusion network replaced by the CGTF transformer with the caption token maskable -> true **video-only mode**
  (separate video-only ensemble used automatically when the caption box is empty).
- Ensemble weights + decision threshold tuned on validation macro-F1; test read once.

**Test results (166 videos, same as v1)**

| Model | Acc | Macro-F1 | AUC |
|---|---|---|---|
| v1 ensemble (original) | 84.3 | 82.0 | 0.938 |
| v1+ ensemble (video + caption) | 86.1 | 83.6 | 0.936 |
| v1+ video-only (no caption) | 80.1 | 76.5 | 0.874 |

Per-model numbers: `models/v1plus/metrics_v1plus.json`. The gain over v1 is 3 more correct test videos and is
within noise on 166 videos; report it as "comparable or slightly better", not as a significant improvement.

**Files added:** `src/v1plus.py`, `src/predictor_v1plus.py`, `scripts/07_v1_improve.py` (experiments, incl. +audio
variant), `scripts/08_build_v1plus.py` (builds the deployed model), `models/v1plus/`, `results/v1_improved/`,
`results/comparison_ieee_fakett.csv`, `results/ieee_related_work.md`.
**Files changed:** `app/server.py` (model choice), `app/static/app.js`, `app/static/index.html` (labels, v2-only fields hidden).
Backups of the old versions: `app/server_v2.py.bak`, `app/static/app_v2.js.bak`, `app/static/index_v2.html.bak`.

**Choosing the model:** default is v1+. Force another with an environment variable before starting:
`set VERIFRAME_MODEL=v2` (Windows) / `export VERIFRAME_MODEL=v2` (Mac/Linux), or `v1` for the original.
