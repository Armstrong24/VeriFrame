"""
Train 5 models on FakeTT text+video features and build a tuned ensemble.
  1. Logistic Regression      (linear baseline)
  2. SVM (RBF)                (strong on embeddings)
  3. LightGBM                 (gradient boosting)
  4. XGBoost                  (gradient boosting)
  5. Cross-Modal Attention Fusion net (PyTorch, 5-seed average)
Ensemble weights + decision threshold are tuned on the validation split,
final numbers are reported on the held-out temporal TEST split.
"""
import json, sys, itertools, warnings
from pathlib import Path
import numpy as np, joblib, torch
torch.set_num_threads(2)
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import tabular_vector, tabular_names, SCALAR_KEYS
from src.fusion_model import CrossModalFusion
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
import lightgbm as lgb, xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "models"; MD.mkdir(exist_ok=True)
D = np.load(ROOT / "data/features.npz", allow_pickle=True)
y = D["label"].astype(int); split = D["split"]
scal_raw = D["scalars"].astype(np.float32)
X = np.stack([tabular_vector(D["frame_emb"][i], D["text_emb"][i], D["sbert"][i],
                             dict(zip(SCALAR_KEYS, scal_raw[i]))) for i in range(len(y))])
tr, va, te = split == "train", split == "val", split == "test"
print(f"samples: train {tr.sum()}  val {va.sum()}  test {te.sum()}  (fake ratio {y.mean():.2f})")


def metrics(yt, p, thr=0.5):
    yp = (p >= thr).astype(int)
    return {"accuracy": accuracy_score(yt, yp), "macro_f1": f1_score(yt, yp, average="macro"),
            "auc": roc_auc_score(yt, p), "precision_fake": precision_score(yt, yp, zero_division=0),
            "recall_fake": recall_score(yt, yp, zero_division=0)}


pos_w = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)

# ---------------------------------------------------------------- tabular models
def make_tab():
    return {
        "logreg": make_pipeline(StandardScaler(), PCA(256, random_state=0),
                                LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced")),
        "svm": make_pipeline(StandardScaler(), PCA(256, random_state=0),
                             SVC(C=2.0, gamma="scale", probability=True, class_weight="balanced", random_state=0)),
        "lightgbm": lgb.LGBMClassifier(n_estimators=700, learning_rate=0.03, num_leaves=15, min_child_samples=15,
                                       colsample_bytree=0.15, subsample=0.8, subsample_freq=1, reg_lambda=2.0,
                                       class_weight="balanced", verbose=-1, random_state=0),
        "xgboost": xgb.XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=4, subsample=0.8,
                                     colsample_bytree=0.15, reg_lambda=2.0, scale_pos_weight=pos_w,
                                     eval_metric="logloss", random_state=0, n_jobs=2),
    }


# ---------------------------------------------------------------- fusion net
smu, ssd = scal_raw[tr].mean(0), scal_raw[tr].std(0) + 1e-6
def T(idx):
    return [torch.tensor(D["frame_emb"][idx]), torch.tensor(D["text_emb"][idx]), torch.tensor(D["sbert"][idx]),
            torch.tensor((scal_raw[idx] - smu) / ssd), torch.tensor(D["frame_sims"][idx])]


def train_fusion(train_idx, val_idx=None, seed=0, epochs=30):
    torch.manual_seed(seed); np.random.seed(seed)
    net = CrossModalFusion(len(SCALAR_KEYS))
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-2)
    Xt = T(train_idx); yt = torch.tensor(y[train_idx], dtype=torch.float32)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, dtype=torch.float32))
    best, best_state = -1, None
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(yt))
        for b in range(0, len(yt), 32):
            ix = perm[b:b + 32]
            batch = [x[ix] for x in Xt]
            batch[0] = batch[0] + 0.02 * torch.randn_like(batch[0])  # feature noise augmentation
            logit, _ = net(*batch)
            loss = lossf(logit, yt[ix]); opt.zero_grad(); loss.backward(); opt.step()
        if val_idx is not None:
            p = predict_fusion([net], val_idx)
            auc = roc_auc_score(y[val_idx], p)
            if auc > best:
                best, best_state = auc, {k: v.clone() for k, v in net.state_dict().items()}
                net.best_epoch = ep + 1
    if best_state:
        net.load_state_dict(best_state)
    return net.eval()


@torch.no_grad()
def predict_fusion(nets, idx):
    Xs = T(idx)
    return np.mean([torch.sigmoid(n.eval()(*Xs)[0]).numpy() for n in nets], 0)


# ---------------------------------------------------------------- stage 1: train on train, eval val/test
itr, iva, ite = np.where(tr)[0], np.where(va)[0], np.where(te)[0]
P_va, P_te, report = {}, {}, {}
for name, m in make_tab().items():
    m.fit(X[itr], y[itr])
    P_va[name], P_te[name] = m.predict_proba(X[iva])[:, 1], m.predict_proba(X[ite])[:, 1]
    print(f"{name:10s} val AUC {roc_auc_score(y[iva], P_va[name]):.3f}  test AUC {roc_auc_score(y[ite], P_te[name]):.3f}")
SEEDS = [0, 1, 2]
nets = [train_fusion(itr, iva, s) for s in SEEDS]
P_va["fusion_net"], P_te["fusion_net"] = predict_fusion(nets, iva), predict_fusion(nets, ite)
BEST_EP = int(np.clip(np.mean([getattr(n, "best_epoch", 15) for n in nets]) * 1.1, 8, 30))
print("fusion best epoch ->", BEST_EP, flush=True)
print(f"fusion_net val AUC {roc_auc_score(y[iva], P_va['fusion_net']):.3f}  test AUC {roc_auc_score(y[ite], P_te['fusion_net']):.3f}")

# ---------------------------------------------------------------- ensemble weight + threshold search on VAL
names = list(P_va)
best = (-1, None, 0.5)
grid = [0, 1, 2, 3]
for w in itertools.product(grid, repeat=len(names)):
    if sum(w) == 0:
        continue
    w = np.array(w) / sum(w)
    pv = sum(w[i] * P_va[n] for i, n in enumerate(names))
    for thr in np.arange(0.3, 0.71, 0.02):
        f = f1_score(y[iva], (pv >= thr).astype(int), average="macro")
        if f > best[0]:
            best = (f, w, float(thr))
_, W, THR = best
ens_te = sum(W[i] * P_te[n] for i, n in enumerate(names))
for n in names:
    report[n] = metrics(y[ite], P_te[n])
report["ENSEMBLE"] = metrics(y[ite], ens_te, THR)
cm = confusion_matrix(y[ite], (ens_te >= THR).astype(int)).tolist()
print("\nTEST results (temporal split):")
for n, r in report.items():
    print(f"  {n:11s} acc {r['accuracy']:.3f}  macroF1 {r['macro_f1']:.3f}  AUC {r['auc']:.3f}")
print("weights", dict(zip(names, np.round(W, 3))), "threshold", THR)

# ---------------------------------------------------------------- stage 2: refit on train+val for deployment
itv = np.concatenate([itr, iva])
final = make_tab()
for name, m in final.items():
    m.fit(X[itv], y[itv])
joblib.dump(final, MD / "tabular_models.joblib")
final_nets = [train_fusion(itv, None, s, epochs=BEST_EP) for s in SEEDS]
for s, n in zip(SEEDS, final_nets):
    torch.save(n.state_dict(), MD / f"fusion_seed{s}.pt")
np.savez(MD / "scalar_norm.npz", mu=smu, sd=ssd)

# feature-group importance from LightGBM
imp = final["lightgbm"].booster_.feature_importance("gain")
from src.features import feature_group
groups = {}
for nm, v in zip(tabular_names(), imp):
    groups[feature_group(nm)] = groups.get(feature_group(nm), 0) + float(v)
tot = sum(groups.values()) or 1
groups = {k: round(v / tot * 100, 1) for k, v in sorted(groups.items(), key=lambda x: -x[1])}

json.dump({"weights": dict(zip(names, W.tolist())), "threshold": THR, "seeds": SEEDS}, open(MD / "ensemble.json", "w"), indent=2)
json.dump({"test_metrics": report, "confusion_matrix_test": cm, "feature_group_importance_pct": groups,
           "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum())},
          open(MD / "metrics.json", "w"), indent=2)
print("saved models ->", MD)
