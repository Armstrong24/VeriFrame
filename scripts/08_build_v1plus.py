"""
Build the deployable VeriFrame v1+ model (models/v1plus/).
Data  : ONLY the original v1 videos (data/features_v1_1299.npz), original split train 962 / val 171 / test 166.
Inputs: v1 features only (no audio, no speech, no bio).  Modes: full (video + caption) and video_only.
All choices (C, ensemble weights, thresholds) are made on VALIDATION; the test set is scored once.
Run:  python scripts/08_build_v1plus.py      (resumable: finished CGTF seeds are cached in models/v1plus/)
"""
import sys, json, itertools, warnings, time
from pathlib import Path
import numpy as np, joblib
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
from src.v1plus import BlockStack, blocks, scal_v2_from_v1, logit, DROP_FULL, DROP_VIDEO
from src.train_v2 import train_cgtf, predict

MD = ROOT / "models/v1plus"; MD.mkdir(parents=True, exist_ok=True)
V = np.load(ROOT / "data/features_v1_1299.npz", allow_pickle=True)
y = V["label"].astype(int); split = V["split"]; N = len(y)
itr, iva, ite = [np.where(split == s)[0] for s in ("train", "val", "test")]
scal = scal_v2_from_v1(V["scalars"])
Z = lambda *s: np.zeros(s, np.float32)
D = {"y": y, "frames": V["frame_emb"].astype(np.float32), "fs_cap": V["frame_sims"].astype(np.float32), "fs_sp": Z(N, 8),
     "cap": np.concatenate([V["text_emb"], V["sbert"]], 1).astype(np.float32), "sp": Z(N, 896), "aud": Z(N, 512), "bio": Z(N, 384),
     "scal": scal, "present": np.tile(np.array([1, 0, 0, 0], np.float32), (N, 1))}
print("train/val/test", len(itr), len(iva), len(ite), flush=True)
pos_w = (y[itr] == 0).sum() / (y[itr] == 1).sum()


def metrics(yt, p, thr):
    yp = (p >= thr).astype(int)
    return {"accuracy": float(accuracy_score(yt, yp)), "macro_f1": float(f1_score(yt, yp, average="macro")),
            "auc": float(roc_auc_score(yt, p))}


def best_thr(p, yv):
    ts = np.arange(0.2, 0.81, 0.02)
    return float(ts[int(np.argmax([f1_score(yv, (p >= t).astype(int), average="macro") for t in ts]))])


def pick(make, X, grid):
    best = None
    for c in grid:
        m = make(c).fit(X[itr], y[itr]); a = roc_auc_score(y[iva], m.predict_proba(X[iva])[:, 1])
        if best is None or a > best[0]:
            best = (a, c, m)
    return best[2], best[1]


mk_lr = lambda C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=4000, class_weight="balanced"))
mk_svm = lambda C: make_pipeline(StandardScaler(), PCA(256, random_state=0), SVC(C=C, gamma="scale", probability=True,
                                                                                   class_weight="balanced", random_state=0))


def tune(P_va, yv):
    names = list(P_va); P = np.stack([P_va[n] for n in names], 1)
    W = np.array([w for w in itertools.product([0, 1, 2], repeat=len(names)) if sum(w)], float); W /= W.sum(1, keepdims=True)
    thr = np.arange(0.3, 0.71, 0.02); best = (-1, None, 0.5)
    pv = P @ W.T; yp = pv[None] >= thr[:, None, None]; yt = yv[None, :, None].astype(bool)
    tp = (yp & yt).sum(1); fp = (yp & ~yt).sum(1); fn = (~yp & yt).sum(1); tn = (~yp & ~yt).sum(1)
    mf = (2 * tp / np.maximum(2 * tp + fp + fn, 1) + 2 * tn / np.maximum(2 * tn + fn + fp, 1)) / 2
    t, k = np.unravel_index(np.argmax(mf), mf.shape)
    return dict(zip(names, W[k].round(4).tolist())), float(thr[t])


SEEDS = [0, 1, 2]
models, cfg, report, cms, preds = {}, {"seeds": SEEDS, "modes": {}}, {}, {}, {}
for mode, drop in (("full", DROP_FULL), ("video_only", DROP_VIDEO)):
    t0 = time.time()
    B = blocks(D["frames"], V["text_emb"], V["sbert"], D["fs_cap"], scal, "full" if mode == "full" else "video")
    X = np.concatenate(list(B.values()), 1)
    sub = lambda Bd, ix: {k: v[ix] for k, v in Bd.items()}
    M = {}
    M["logreg"], c_lr = pick(mk_lr, X, [0.003, 0.01, 0.03, 0.1, 0.3])
    M["svm"], c_svm = pick(mk_svm, X, [0.5, 1, 2, 4])
    M["lightgbm"] = lgb.LGBMClassifier(n_estimators=700, learning_rate=0.03, num_leaves=15, min_child_samples=15, colsample_bytree=0.15,
                                       subsample=0.8, subsample_freq=1, reg_lambda=2.0, class_weight="balanced", verbose=-1,
                                       random_state=0, n_jobs=2).fit(X[itr], y[itr])
    M["xgboost"] = xgb.XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.15,
                                     reg_lambda=2.0, scale_pos_weight=pos_w, eval_metric="logloss", random_state=0, n_jobs=2).fit(X[itr], y[itr])
    P_va = {n: m.predict_proba(X[iva])[:, 1] for n, m in M.items()}
    P_te = {n: m.predict_proba(X[ite])[:, 1] for n, m in M.items()}
    M["block_stack"] = BlockStack().fit(sub(B, itr), y[itr], sub(B, iva), y[iva])
    P_va["block_stack"] = M["block_stack"].predict_proba(sub(B, iva))[:, 1]; P_te["block_stack"] = M["block_stack"].predict_proba(sub(B, ite))[:, 1]
    va_s, te_s = [], []
    for s in SEEDS:
        ck = MD / f"cgtf_{mode}_seed{s}.pt"
        if ck.exists():
            c = torch.load(ck, map_location="cpu", weights_only=False)
            from src.v2 import CGTF
            n = CGTF(); n.load_state_dict(c["state"]); n.mu, n.sd = c["mu"], c["sd"]; n.eval()
        else:
            n = train_cgtf(D, itr, iva, seed=s, mdrop=(mode == "full"), ablate=drop,
                           select_on=("full", "video") if mode == "full" else ("full",))
            torch.save({"state": n.state_dict(), "mu": n.mu, "sd": n.sd}, ck)
        va_s.append(predict(n, D, iva, drop=drop)); te_s.append(predict(n, D, ite, drop=drop))
    P_va["fusion_cgtf"], P_te["fusion_cgtf"] = np.mean(va_s, 0), np.mean(te_s, 0)
    W, THR = tune(P_va, y[iva])
    ens_te = sum(W[n] * P_te[n] for n in W)
    report[mode] = {n: metrics(y[ite], P_te[n], best_thr(P_va[n], y[iva])) for n in P_te}
    report[mode]["ENSEMBLE"] = metrics(y[ite], ens_te, THR)
    cms[mode] = confusion_matrix(y[ite], (ens_te >= THR).astype(int)).tolist()
    cfg["modes"][mode] = {"weights": W, "threshold": THR, "logreg_C": c_lr, "svm_C": c_svm}
    models[mode] = {k: v for k, v in M.items()}
    preds[mode] = {**P_te, "ENSEMBLE": ens_te}
    print(f"\n=== {mode} ({time.time() - t0:.0f}s)  weights {W} thr {THR}")
    for n, r in report[mode].items():
        print(f"  {n:12s} acc {r['accuracy']*100:5.1f}  F1 {r['macro_f1']*100:5.1f}  AUC {r['auc']:.3f}", flush=True)

joblib.dump(models, MD / "tabular_models_v1plus.joblib", compress=3)
json.dump(cfg, open(MD / "ensemble_v1plus.json", "w"), indent=2)
json.dump({"version": 3, "name": "VeriFrame v1+", "test_metrics": report, "confusion_matrix_test": cms,
           "n_train": len(itr), "n_val": len(iva), "n_test": len(ite),
           "note": "Original v1 videos (1,299) and split only; v1 features only; thresholds/weights tuned on validation."},
          open(MD / "metrics_v1plus.json", "w"), indent=2)
np.savez(ROOT / "results/test_predictions_v1plus.npz", vid=V["vid"][ite], y=y[ite],
         **{f"{m}__{n}": p for m in preds for n, p in preds[m].items()}, thr_full=cfg["modes"]["full"]["threshold"],
         thr_video=cfg["modes"]["video_only"]["threshold"])
print("saved ->", MD)
