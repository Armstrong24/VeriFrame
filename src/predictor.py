"""Loads the trained ensemble and analyzes a single video + caption."""
import json, base64
from pathlib import Path
import numpy as np, joblib, torch, cv2
from .features import extract, tabular_vector, tabular_names, feature_group, SCALAR_KEYS
from .fusion_model import CrossModalFusion

MD = Path(__file__).resolve().parents[1] / "models"


def _load_tabular():
    """Load the saved sklearn/LightGBM/XGBoost models. If your installed library versions can't
    unpickle them, retrain automatically from data/features.npz (takes a few minutes on CPU)."""
    try:
        return joblib.load(MD / "tabular_models.joblib")
    except Exception as e:
        import subprocess, sys
        print(f"[VeriFrame] Saved models incompatible with your library versions ({e}). Retraining...")
        subprocess.run([sys.executable, str(MD.parent / "scripts" / "03_train.py")], check=True)
        return joblib.load(MD / "tabular_models.joblib")


class Predictor:
    def __init__(self):
        self.tab = _load_tabular()
        self.ens = json.load(open(MD / "ensemble.json"))
        nz = np.load(MD / "scalar_norm.npz"); self.mu, self.sd = nz["mu"], nz["sd"]
        self.nets = []
        for s in self.ens["seeds"]:
            n = CrossModalFusion(len(SCALAR_KEYS)); n.load_state_dict(torch.load(MD / f"fusion_seed{s}.pt", map_location="cpu"))
            self.nets.append(n.eval())
        self.names = tabular_names()
        self.metrics = json.load(open(MD / "metrics.json")) if (MD / "metrics.json").exists() else {}

    @torch.no_grad()
    def analyze(self, video_path, caption, verified=0):
        f = extract(video_path, caption, verified)
        s = f["scalars"]
        x = tabular_vector(f["frame_emb"], f["text_emb"], f["sbert"], s)[None]
        probs = {n: float(m.predict_proba(x)[0, 1]) for n, m in self.tab.items()}
        sc = torch.tensor(((np.array([s[k] for k in SCALAR_KEYS], np.float32) - self.mu) / self.sd)[None])
        inp = [torch.tensor(f["frame_emb"][None]), torch.tensor(f["text_emb"][None]), torch.tensor(f["sbert"][None]),
               sc.float(), torch.tensor(f["frame_sims"][None])]
        outs = [n(*inp) for n in self.nets]
        probs["fusion_net"] = float(np.mean([torch.sigmoid(o[0]).item() for o in outs]))
        attn = np.mean([o[1].numpy()[0] for o in outs], 0)
        w = self.ens["weights"]
        p = sum(w[k] * probs[k] for k in w)
        thr = self.ens["threshold"]

        # explanation: LightGBM SHAP-style contributions grouped by modality
        contrib = self.tab["lightgbm"].booster_.predict(x, pred_contrib=True)[0][:-1]
        groups = {}
        for nm, c in zip(self.names, contrib):
            g = feature_group(nm); groups[g] = groups.get(g, 0.0) + float(c)
        top_scalar = sorted([(nm, float(c)) for nm, c in zip(self.names, contrib) if nm in SCALAR_KEYS],
                            key=lambda t: -abs(t[1]))[:6]

        thumbs = []
        for i, fr in enumerate(f["frames"]):
            h, wd = fr.shape[:2]; sc_ = 220 / max(h, wd)
            small = cv2.resize(fr, (int(wd * sc_), int(h * sc_)))
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
            thumbs.append({"img": "data:image/jpeg;base64," + base64.b64encode(buf).decode(),
                           "consistency": float(f["frame_sims"][i]), "attention": float(attn[i])})
        conf = p if p >= thr else 1 - p
        return {
            "verdict": "FAKE" if p >= thr else "REAL", "fake_probability": float(p), "threshold": thr,
            "confidence": float(conf), "model_probs": probs, "weights": w,
            "modality_contrib": groups, "top_signals": top_scalar,
            "frames": thumbs, "video_stats": {k: float(s[k]) for k in SCALAR_KEYS[:13]},
            "text_stats": {k: float(s[k]) for k in SCALAR_KEYS[13:23]},
            "consistency": {k: float(s[k]) for k in SCALAR_KEYS[23:30]},
        }
