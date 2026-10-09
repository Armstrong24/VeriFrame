"""
VeriFrame v1+ : try to improve accuracy using ONLY the original v1 videos (1,299) and the original
v1 temporal split (train 962 / val 171 / test 166). Adds a video-only mode.

Two feature settings on the SAME videos:
  A = v1 features only (CLIP frames, caption CLIP+MiniLM, v1 scalars)
  B = A + audio/speech from the same videos (Whisper transcript, CLAP, publisher bio)
Every choice (C values, ensemble weights, threshold) is made on VALIDATION only; test is read once.
Outputs -> results/v1_improved/
"""
import sys, json, itertools, warnings, time
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import torch; torch.set_num_threads(2)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import lightgbm as lgb, xgboost as xgb
from src.data_v2 import load_all
from src.v2 import SCALARS_V2, TEXT_IDX
from src.features import VIDEO_STAT_KEYS, TEXT_STAT_KEYS
from src.train_v2 import train_cgtf, predict

OUT = ROOT / "results/v1_improved"; OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "cache"; CACHE.mkdir(exist_ok=True)

D0 = load_all(); V = np.load(ROOT / "data/features_v1_1299.npz", allow_pickle=True)
pos = {v: i for i, v in enumerate(D0["vid"])}; sel = np.array([pos[v] for v in V["vid"]])
D = {k: (v[sel] if isinstance(v, np.ndarray) and len(v) == len(D0["vid"]) else v) for k, v in D0.items()}
D["split"] = V["split"]; y = D["y"]
assert (y == V["label"]).all()
itr, iva, ite = [np.where(D["split"] == s)[0] for s in ("train", "val", "test")]
print("train/val/test", len(itr), len(iva), len(ite), flush=True)

S = D["scal"]; K = {k: i for i, k in enumerate(SCALARS_V2)}
fr = D["frames"]
vis = np.concatenate([fr.mean(1), fr.std(1), fr.max(1), np.abs(np.diff(fr, axis=1)).mean(1)], 1)
vstat = S[:, [K[k] for k in VIDEO_STAT_KEYS + ["temporal_emb_change_mean", "temporal_emb_change_max", "frame_diversity"]]]
cap = np.concatenate([D["text_emb"], D["sbert"], fr.mean(1) * D["text_emb"], D["fs_cap"],
                      S[:, [K[k] for k in TEXT_STAT_KEYS + ["clip_sim_mean", "clip_sim_max", "clip_sim_min", "clip_sim_std", "user_verified"]]]], 1)
sp_keys = ["cap_speech_sim", "speech_frame_sim_mean", "speech_frame_sim_max", "speech_frame_sim_min",
           "n_words", "words_per_sec", "has_speech", "speech_ratio"]
sp_v = np.concatenate([D["sp_clip"], D["sp_sbert"], fr.mean(1) * D["sp_clip"], D["fs_sp"],
                       S[:, [K[k] for k in sp_keys if k != "cap_speech_sim"]]], 1)
from src.audio_features import AUDIO_SCALAR_KEYS
aud = np.concatenate([D["aud"], S[:, [K[k] for k in AUDIO_SCALAR_KEYS if k not in sp_keys]]], 1)
bio = np.concatenate([D["bio"], D["present"][:, 3:4]], 1)
capx = np.concatenate([S[:, [K["cap_speech_sim"], K["cap_audio_sim"]]]], 1)   # caption<->speech/audio consistency

CONFIGS = {
    "A_full":  {"blocks": {"vis": vis, "vstat": vstat, "cap": cap}, "drop": frozenset({"speech", "audio", "bio"})},
    "A_video": {"blocks": {"vis": vis, "vstat": vstat}, "drop": frozenset({"speech", "audio", "bio", "caption"})},
    "B_full":  {"blocks": {"vis": vis, "vstat": vstat, "cap": np.concatenate([cap, capx], 1), "sp": sp_v, "aud": aud, "bio": bio},
                "drop": frozenset()},
    "B_video": {"blocks": {"vis": vis, "vstat": vstat, "sp": sp_v, "aud": aud}, "drop": frozenset({"caption", "bio"})},
}
pos_w = (y[itr] == 0).sum() / (y[itr] == 1).sum()


def metrics(yt, p, thr=0.5):
    yp = (p >= thr).astype(int)
    return {"accuracy": round(accuracy_score(yt, yp) * 100, 1), "macro_f1": round(f1_score(yt, yp, average="macro") * 100, 1),
            "auc": round(roc_auc_score(yt, p), 3)}


def best_thr(p, yv):
    ts = np.arange(0.2, 0.81, 0.02)
    f = [f1_score(yv, (p >= t).astype(int), average="macro") for t in ts]
    return float(ts[int(np.argmax(f))])


def lr_pick(X):
    """LogReg with C picked on val AUC."""
    best = None
    for C in [0.003, 0.01, 0.03, 0.1, 0.3]:
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=4000, class_weight="balanced")).fit(X[itr], y[itr])
        a = roc_auc_score(y[iva], m.predict_proba(X[iva])[:, 1])
        if best is None or a > best[0]:
            best = (a, C, m)
    return best[2], best[1]


def svm_pick(X):
    best = None
    for C in [0.5, 1, 2, 4]:
        m = make_pipeline(StandardScaler(), PCA(min(256, X.shape[1]), random_state=0),
                          SVC(C=C, gamma="scale", probability=True, class_weight="balanced", random_state=0)).fit(X[itr], y[itr])
        a = roc_auc_score(y[iva], m.predict_proba(X[iva])[:, 1])
        if best is None or a > best[0]:
            best = (a, C, m)
    return best[2], best[1]


def block_stack(blocks):
    """Late fusion: one regularised LogReg per modality block, then a meta LogReg on their out-of-fold scores."""
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    oof = np.zeros((len(itr), len(blocks))); va = np.zeros((len(iva), len(blocks))); te = np.zeros((len(ite), len(blocks)))
    for j, (nm, X) in enumerate(blocks.items()):
        _, C = lr_pick(X)
        mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=4000, class_weight="balanced"))
        for a, b in skf.split(itr, y[itr]):
            oof[b, j] = mk().fit(X[itr][a], y[itr][a]).predict_proba(X[itr][b])[:, 1]
        m = mk().fit(X[itr], y[itr]); va[:, j] = m.predict_proba(X[iva])[:, 1]; te[:, j] = m.predict_proba(X[ite])[:, 1]
    lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    meta = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(lg(oof), y[itr])
    return meta.predict_proba(lg(va))[:, 1], meta.predict_proba(lg(te))[:, 1], dict(zip(blocks, meta.coef_[0].round(2).tolist()))


def tune(P_va, yv):
    names = list(P_va); P = np.stack([P_va[n] for n in names], 1)
    W = np.array([w for w in itertools.product([0, 1, 2], repeat=len(names)) if sum(w)], float); W /= W.sum(1, keepdims=True)
    thr = np.arange(0.3, 0.71, 0.02); best = (-1, None, 0.5)
    for c in range(0, len(W), 1024):
        pv = P @ W[c:c + 1024].T; yp = pv[None] >= thr[:, None, None]; yt = yv[None, :, None].astype(bool)
        tp = (yp & yt).sum(1); fp = (yp & ~yt).sum(1); fn = (~yp & yt).sum(1); tn = (~yp & ~yt).sum(1)
        mf = (2 * tp / np.maximum(2 * tp + fp + fn, 1) + 2 * tn / np.maximum(2 * tn + fn + fp, 1)) / 2
        t, k = np.unravel_index(np.argmax(mf), mf.shape)
        if mf[t, k] > best[0] + 1e-12:
            best = (mf[t, k], W[c + k], float(thr[t]))
    return dict(zip(names, best[1].round(3).tolist())), best[2]


SEEDS = [0, 1, 2]
nets = {}
results, preds, info = {}, {}, {}
for cfg_name, cfg in CONFIGS.items():
    t0 = time.time(); blocks = cfg["blocks"]; X = np.concatenate(list(blocks.values()), 1).astype(np.float32)
    cf = CACHE / f"{cfg_name}.npz"
    if cf.exists():
        z = np.load(cf, allow_pickle=True); P_va = z["va"].item(); P_te = z["te"].item(); info[cfg_name] = z["info"].item()
    else:
        P_va, P_te, inf = {}, {}, {}
        m, C = lr_pick(X); P_va["logreg"], P_te["logreg"] = m.predict_proba(X[iva])[:, 1], m.predict_proba(X[ite])[:, 1]; inf["logreg_C"] = C
        m, C = svm_pick(X); P_va["svm"], P_te["svm"] = m.predict_proba(X[iva])[:, 1], m.predict_proba(X[ite])[:, 1]; inf["svm_C"] = C
        m = lgb.LGBMClassifier(n_estimators=700, learning_rate=0.03, num_leaves=15, min_child_samples=15, colsample_bytree=0.15,
                               subsample=0.8, subsample_freq=1, reg_lambda=2.0, class_weight="balanced", verbose=-1, random_state=0, n_jobs=2).fit(X[itr], y[itr])
        P_va["lightgbm"], P_te["lightgbm"] = m.predict_proba(X[iva])[:, 1], m.predict_proba(X[ite])[:, 1]
        m = xgb.XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.15, reg_lambda=2.0,
                              scale_pos_weight=pos_w, eval_metric="logloss", random_state=0, n_jobs=2).fit(X[itr], y[itr])
        P_va["xgboost"], P_te["xgboost"] = m.predict_proba(X[iva])[:, 1], m.predict_proba(X[ite])[:, 1]
        if len(blocks) > 1:
            P_va["block_stack"], P_te["block_stack"], inf["block_weights"] = block_stack(blocks)
        np.savez(cf, va=P_va, te=P_te, info=inf); info[cfg_name] = inf
    # CGTF fusion net (same architecture as v2, inputs masked to this config)
    va_s, te_s = [], []
    for s in SEEDS:
        cc = CACHE / f"cgtf_{cfg_name}_s{s}.npz"
        if not cc.exists():
            train_drop = cfg["drop"] - {"caption"} if cfg_name.endswith("video") else cfg["drop"]
            n = train_cgtf(D, itr, iva, seed=s, ablate=frozenset(train_drop),
                           select_on=("video",) if cfg_name.endswith("video") else ("full", "video"))
            np.savez(cc, va=predict(n, D, iva, drop=cfg["drop"]), te=predict(n, D, ite, drop=cfg["drop"]))
            torch.save({"state": n.state_dict(), "mu": n.mu, "sd": n.sd, "drop": sorted(cfg["drop"])}, CACHE / f"cgtf_{cfg_name}_s{s}.pt")
        z = np.load(cc); va_s.append(z["va"]); te_s.append(z["te"])
    P_va["fusion_cgtf"], P_te["fusion_cgtf"] = np.mean(va_s, 0), np.mean(te_s, 0)
    W, THR = tune(P_va, y[iva])
    ens_va = sum(W[n] * P_va[n] for n in W); ens = sum(W[n] * P_te[n] for n in W)
    avg = np.mean([P_te[n] for n in P_te], 0); avg_thr = best_thr(np.mean([P_va[n] for n in P_va], 0), y[iva])
    r = {n: metrics(y[ite], P_te[n], best_thr(P_va[n], y[iva])) for n in P_te}
    r["ENSEMBLE (val-tuned weights)"] = metrics(y[ite], ens, THR)
    r["ENSEMBLE (equal average)"] = metrics(y[ite], avg, avg_thr)
    r["_val_ensemble"] = metrics(y[iva], ens_va, THR)
    results[cfg_name] = r; info[cfg_name].update(weights=W, threshold=THR)
    preds[cfg_name] = {**P_te, "ENSEMBLE": ens}
    print(f"\n=== {cfg_name} ({time.time() - t0:.0f}s)")
    for n, m_ in r.items():
        print(f"  {n:30s} acc {m_['accuracy']:5.1f}  F1 {m_['macro_f1']:5.1f}  AUC {m_['auc']:.3f}")
    print("  ", info[cfg_name], flush=True)

json.dump({"results": results, "info": info, "note": "v1 videos only (1,299), v1 split; thresholds/weights tuned on val"},
          open(OUT / "metrics_v1_improved.json", "w"), indent=2, default=float)
np.savez(OUT / "test_predictions.npz", vid=D["vid"][ite], y=y[ite], **{f"{c}__{n}": p for c in preds for n, p in preds[c].items()})
print("done")
