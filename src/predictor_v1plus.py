"""VeriFrame v1+ predictor (models/v1plus/). Uses only v1 inputs: CLIP frames + caption.
Empty caption -> VIDEO-ONLY ensemble (no text at all). No Whisper / CLAP needed, so it is light on RAM."""
import json, base64
from pathlib import Path
import numpy as np, joblib, torch, cv2
from .features import extract, SCALAR_KEYS
from .v2 import CGTF, SCALARS_V2
from .train_v2 import apply_mask
from .v1plus import blocks, scal_v2_from_v1, column_names, DROP_FULL, DROP_VIDEO

MD = Path(__file__).resolve().parents[1] / "models" / "v1plus"


class PredictorV1Plus:
    def __init__(self):
        self.tab = joblib.load(MD / "tabular_models_v1plus.joblib")
        # XGBoost is stored in its own version-independent JSON format (pickles break across xgboost versions)
        for mode in self.tab:
            fx = MD / f"xgboost_{mode}.json"
            if fx.exists() and "xgboost" not in self.tab[mode]:
                try:
                    import xgboost as xgb
                    m = xgb.XGBClassifier(); m.load_model(fx); self.tab[mode]["xgboost"] = m
                except Exception as e:                      # weight is 0 in both ensembles, so it is optional
                    print(f"[VeriFrame] XGBoost ({mode}) not loaded: {e}")
        self.cfg = json.load(open(MD / "ensemble_v1plus.json"))
        self.nets = {}
        for mode in ("full", "video_only"):
            self.nets[mode] = []
            for s in self.cfg["seeds"]:
                ck = torch.load(MD / f"cgtf_{mode}_seed{s}.pt", map_location="cpu", weights_only=False)
                n = CGTF(); n.load_state_dict(ck["state"]); n.mu, n.sd = ck["mu"], ck["sd"]; self.nets[mode].append(n.eval())
        self.metrics = json.load(open(MD / "metrics_v1plus.json"))
        # feature-group importance (LightGBM gain) for the dashboard's model card
        nm = column_names("full"); gain = self.tab["full"]["lightgbm"].booster_.feature_importance("gain"); g = {}
        for (_, grp), v in zip(nm, gain):
            g[grp] = g.get(grp, 0) + float(v)
        tot = sum(g.values()) or 1
        self.metrics["feature_group_importance_pct"] = {k: round(100 * v / tot, 1) for k, v in sorted(g.items(), key=lambda t: -t[1])}

    @torch.no_grad()
    def analyze(self, video_path, caption="", verified=0, bio=""):
        has_cap = bool((caption or "").strip())
        mode = "full" if has_cap else "video_only"
        f = extract(video_path, caption if has_cap else "", verified if has_cap else 0)
        scal = scal_v2_from_v1(np.array([f["scalars"][k] for k in SCALAR_KEYS], np.float32))
        B = blocks(f["frame_emb"][None], f["text_emb"][None], f["sbert"][None], f["frame_sims"][None], scal,
                   "full" if has_cap else "video")
        x = np.concatenate(list(B.values()), 1)
        probs = {n: float(m.predict_proba(B if n == "block_stack" else x)[0, 1]) for n, m in self.tab[mode].items()}
        drop = DROP_FULL if has_cap else DROP_VIDEO
        T = lambda q: torch.tensor(np.ascontiguousarray(q), dtype=torch.float32)
        cg = []
        for n in self.nets[mode]:
            s = (scal - n.mu) / n.sd
            s, fsc, fss, p = apply_mask(s, f["frame_sims"][None], np.zeros((1, 8), np.float32), np.array([[1, 0, 0, 0]], np.float32), drop)
            cg.append(torch.sigmoid(n(T(f["frame_emb"][None]), T(fsc), T(fss), T(np.concatenate([f["text_emb"], f["sbert"]])[None]),
                                      T(np.zeros((1, 896))), T(np.zeros((1, 512))), T(np.zeros((1, 384))), T(s), T(p))).item())
        probs["fusion_cgtf"] = float(np.mean(cg))
        w = self.cfg["modes"][mode]["weights"]; thr = self.cfg["modes"][mode]["threshold"]
        pf = float(sum(w[k] * probs[k] for k in w))

        names = column_names("full" if has_cap else "video")
        contrib = self.tab[mode]["lightgbm"].booster_.predict(x, pred_contrib=True)[0][:-1]
        groups, sig = {}, []
        for (k, grp), c in zip(names, contrib):
            groups[grp] = groups.get(grp, 0.0) + float(c)
            if k in SCALARS_V2:
                sig.append((k, float(c)))
        top = sorted(sig, key=lambda t: -abs(t[1]))[:6]

        thumbs = []
        for i, fr in enumerate(f["frames"]):
            h, wd = fr.shape[:2]; sc_ = 220 / max(h, wd)
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(cv2.resize(fr, (int(wd * sc_), int(h * sc_))), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
            thumbs.append({"img": "data:image/jpeg;base64," + base64.b64encode(buf).decode(),
                           "consistency": float(f["frame_sims"][i]) if has_cap else None, "attention": 1 / len(f["frames"])})
        vs = {k: float(f["scalars"][k]) for k in SCALAR_KEYS[:13]}
        ts = {k: float(f["scalars"][k]) for k in SCALAR_KEYS[13:23]} if has_cap else {}
        cs = {k: float(f["scalars"][k]) for k in SCALAR_KEYS[23:30]} if has_cap else {k: float(f["scalars"][k]) for k in SCALAR_KEYS[27:30]}
        return {
            "version": 3, "mode": mode, "verdict": "FAKE" if pf >= thr else "REAL", "fake_probability": pf,
            "threshold": thr, "confidence": float(pf if pf >= thr else 1 - pf), "model_probs": probs, "weights": w,
            "modality_contrib": groups, "top_signals": top, "frames": thumbs, "video_stats": vs, "text_stats": ts,
            "consistency": cs, "transcript": "",
        }
