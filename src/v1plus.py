"""VeriFrame v1+ : same videos and same features as v1 (CLIP frames + caption CLIP/MiniLM + v1 scalars),
with better training (val-tuned C, block-wise late fusion, CGTF fusion net) and a video-only mode.
No audio / speech / publisher bio are used, so the dashboard does not need Whisper or CLAP."""
import numpy as np
from .features import VIDEO_STAT_KEYS, TEXT_STAT_KEYS, SCALAR_KEYS
from .v2 import SCALARS_V2

K = {k: i for i, k in enumerate(SCALARS_V2)}
VSTAT_KEYS = VIDEO_STAT_KEYS + ["temporal_emb_change_mean", "temporal_emb_change_max", "frame_diversity"]
CAPSTAT_KEYS = TEXT_STAT_KEYS + ["clip_sim_mean", "clip_sim_max", "clip_sim_min", "clip_sim_std", "user_verified"]
DROP_FULL = frozenset({"speech", "audio", "bio"})          # CGTF modalities never used by v1+
DROP_VIDEO = DROP_FULL | {"caption"}


def scal_v2_from_v1(v1_scalars):
    """[N, 31] v1 scalars -> [N, len(SCALARS_V2)] with every audio/speech slot fixed at 0."""
    v1_scalars = np.atleast_2d(v1_scalars)
    s = np.zeros((len(v1_scalars), len(SCALARS_V2)), np.float32)
    s[:, :len(SCALAR_KEYS)] = v1_scalars
    return s


def blocks(frames, text_emb, sbert, frame_sims, scal, mode="full"):
    """frames [N,8,512]; scal [N, len(SCALARS_V2)] -> dict of feature blocks (same order every time)."""
    vis = np.concatenate([frames.mean(1), frames.std(1), frames.max(1), np.abs(np.diff(frames, axis=1)).mean(1)], 1)
    out = {"vis": vis, "vstat": scal[:, [K[k] for k in VSTAT_KEYS]]}
    if mode == "full":
        out["cap"] = np.concatenate([text_emb, sbert, frames.mean(1) * text_emb, frame_sims,
                                     scal[:, [K[k] for k in CAPSTAT_KEYS]]], 1)
    return {k: v.astype(np.float32) for k, v in out.items()}


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class BlockStack:
    """Late fusion: one regularised LogReg per feature block + a meta LogReg on their out-of-fold scores."""
    GRID = [0.003, 0.01, 0.03, 0.1, 0.3]

    def fit(self, B, y, B_val, y_val):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score
        mk = lambda C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=4000, class_weight="balanced"))
        names = list(B); oof = np.zeros((len(y), len(names))); self.models = {}
        for j, nm in enumerate(names):
            C = max(self.GRID, key=lambda c: roc_auc_score(y_val, mk(c).fit(B[nm], y).predict_proba(B_val[nm])[:, 1]))
            for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(B[nm], y):
                oof[b, j] = mk(C).fit(B[nm][a], y[a]).predict_proba(B[nm][b])[:, 1]
            self.models[nm] = mk(C).fit(B[nm], y)
        self.meta = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(logit(oof), y)
        return self

    def predict_proba(self, B):
        z = np.stack([self.models[nm].predict_proba(B[nm])[:, 1] for nm in self.models], 1)
        return self.meta.predict_proba(logit(z))


def column_names(mode="full"):
    """(feature name, group) for every column of np.concatenate(blocks(...).values(), 1)."""
    cols = [(f"vis_{i}", "Visual (CLIP frames)") for i in range(2048)]
    cols += [(k, "Editing & video stats" if k in VIDEO_STAT_KEYS else "Temporal consistency") for k in VSTAT_KEYS]
    if mode == "full":
        cols += [(f"cap_{i}", "Caption (text)") for i in range(896)]
        cols += [(f"vt_{i}", "Caption-video consistency") for i in range(512)]
        cols += [(f"frame_sim_{i}", "Caption-video consistency") for i in range(8)]
        cols += [(k, "Caption style" if k in TEXT_STAT_KEYS else ("Publisher" if k == "user_verified" else "Caption-video consistency"))
                 for k in CAPSTAT_KEYS]
    return cols
