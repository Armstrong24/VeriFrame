"""Extract multimodal features for every downloaded FakeTT video -> data/features.npz"""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import extract, SCALAR_KEYS

ROOT = Path(__file__).resolve().parents[1]
meta = [json.loads(l) for l in open(ROOT / "data/fakett_meta.jsonl", encoding="utf-8")]
vdir = ROOT / "videos"
out = ROOT / "data/features.npz"
cache = ROOT / "data/feat_cache"; cache.mkdir(exist_ok=True)

t0 = time.time(); done = 0
for i, m in enumerate(meta):
    vid = m["video_id"]; p = vdir / f"{vid}.mp4"; c = cache / f"{vid}.npz"
    if c.exists() or not p.exists():
        continue
    try:
        f = extract(p, m.get("description", ""), m.get("user_certify", 0))
        np.savez(c, frame_emb=f["frame_emb"], text_emb=f["text_emb"], sbert=f["sbert"],
                 frame_sims=f["frame_sims"], scalars=np.array([f["scalars"][k] for k in SCALAR_KEYS], np.float32))
        done += 1
        if done % 25 == 0:
            print(f"{done} videos  {time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        print("skip", vid, e, flush=True)

# pack
rows = {k: [] for k in ["vid", "label", "split", "frame_emb", "text_emb", "sbert", "frame_sims", "scalars"]}
for m in meta:
    c = cache / f"{m['video_id']}.npz"
    if not c.exists():
        continue
    d = np.load(c)
    rows["vid"].append(m["video_id"]); rows["label"].append(1 if m["annotation"] == "fake" else 0)
    rows["split"].append(m["split"])
    for k in ["frame_emb", "text_emb", "sbert", "frame_sims", "scalars"]:
        rows[k].append(d[k])
np.savez_compressed(out, **{k: np.array(v) for k, v in rows.items()})
print("packed", len(rows["vid"]), "samples ->", out)
