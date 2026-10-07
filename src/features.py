"""
Shared feature extraction for training AND the web dashboard.
Text + Video multimodal features for fake/manipulated short-video detection.

Modalities
  - Video  : 8 uniformly sampled frames -> CLIP ViT-B/32 image embeddings
             + temporal/editing stats (shot cuts, frame-to-frame change, duration, fps, resolution)
  - Text   : caption -> CLIP text embedding + MiniLM sentence embedding + style stats
  - Cross  : CLIP similarity between caption and every frame (text-video consistency)
"""
import re
import cv2
import numpy as np
import torch

N_FRAMES = 8
CLIP_NAME = "openai/clip-vit-base-patch32"
SBERT_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_models = {}


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_models():
    if not _models:
        from transformers import CLIPModel, CLIPProcessor
        from sentence_transformers import SentenceTransformer
        dev = _device()
        _models["clip"] = CLIPModel.from_pretrained(CLIP_NAME).to(dev).eval()
        _models["proc"] = CLIPProcessor.from_pretrained(CLIP_NAME)
        _models["sbert"] = SentenceTransformer(SBERT_NAME, device=dev)
    return _models


# ---------------------------------------------------------------- video
def read_video(path, n_frames=N_FRAMES, n_cut_samples=32, max_seconds=120):
    """Decode once with PyAV (multi-threaded, skips non-reference frames) and sample frames.
    Returns (frames_rgb list[n_frames], stats dict)."""
    import av
    c = av.open(str(path))
    st = c.streams.video[0]
    st.thread_type = "AUTO"
    st.codec_context.skip_frame = "NONREF"
    fps = float(st.average_rate or st.guessed_rate or 25)
    w, h = st.codec_context.width, st.codec_context.height
    if st.duration:
        duration = float(st.duration * st.time_base)
    elif c.duration:
        duration = c.duration / 1e6
    else:
        duration = (st.frames or 0) / fps
    if duration <= 0:
        raise ValueError("Could not read video duration")
    span = min(duration, max_seconds)
    t_frames = np.linspace(0, span * 0.98, n_frames)
    t_cuts = np.linspace(0, span * 0.98, n_cut_samples)
    targets = sorted(set(np.round(np.concatenate([t_frames, t_cuts]), 3).tolist()))
    got, ti = {}, 0
    for fr in c.decode(st):
        if fr.time is None:
            continue
        if fr.time > span + 0.5:
            break
        while ti < len(targets) and fr.time >= targets[ti]:
            got[targets[ti]] = fr.to_ndarray(format="rgb24"); ti += 1
        if ti >= len(targets):
            break
    c.close()
    if not got:
        raise ValueError("No frames decoded")
    keys = np.array(sorted(got))

    def nearest(t):
        return got[float(keys[np.abs(keys - t).argmin()])]
    fr_list = [nearest(t) for t in np.round(t_frames, 3)]

    hists, smalls = [], []
    for t in np.round(t_cuts, 3):
        fr = nearest(t)
        hsv = cv2.cvtColor(cv2.resize(fr, (160, 160)), cv2.COLOR_RGB2HSV)
        hst = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256]); cv2.normalize(hst, hst)
        hists.append(hst)
        smalls.append(cv2.cvtColor(cv2.resize(fr, (64, 64)), cv2.COLOR_RGB2GRAY).astype(np.float32))
    corr = [cv2.compareHist(hists[i - 1], hists[i], cv2.HISTCMP_CORREL) for i in range(1, len(hists))]
    diffs = [float(np.abs(smalls[i] - smalls[i - 1]).mean()) for i in range(1, len(smalls))]
    cuts = int(sum(cc < 0.6 for cc in corr))
    brightness = float(np.mean([f.mean() for f in fr_list]))
    sharp = float(np.mean([cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var() for f in fr_list]))
    stats = {
        "duration_s": duration, "fps": fps, "width": w, "height": h,
        "aspect": w / max(h, 1), "is_vertical": float(h > w),
        "shot_cuts": cuts, "cuts_per_10s": cuts / max(span, 1e-3) * 10,
        "motion_mean": float(np.mean(diffs)) if diffs else 0.0,
        "motion_max": float(np.max(diffs)) if diffs else 0.0,
        "motion_std": float(np.std(diffs)) if diffs else 0.0,
        "brightness": brightness, "sharpness": float(np.log1p(sharp)),
    }
    return fr_list, stats


# ---------------------------------------------------------------- text
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")


def text_stats(text):
    t = text or ""
    words = t.split()
    letters = [c for c in t if c.isalpha()]
    return {
        "txt_len": len(t), "txt_words": len(words),
        "hashtags": t.count("#"), "mentions": t.count("@"),
        "exclaims": t.count("!"), "questions": t.count("?"),
        "caps_ratio": (sum(c.isupper() for c in letters) / len(letters)) if letters else 0.0,
        "emojis": len(EMOJI_RE.findall(t)),
        "has_url": float("http" in t or "www." in t),
        "clickbait": float(bool(re.search(r"\b(shocking|breaking|you won'?t believe|must watch|exposed|truth|they don'?t want|viral|omg|wow|share)\b", t.lower()))),
    }


def clean_caption(text):
    t = re.sub(r"#\w+", lambda m: m.group(0)[1:], text or "")  # keep hashtag words
    t = re.sub(r"\s+", " ", t).strip()
    return t if t else "a video"


# ---------------------------------------------------------------- full
VIDEO_STAT_KEYS = ["duration_s", "fps", "width", "height", "aspect", "is_vertical", "shot_cuts",
                   "cuts_per_10s", "motion_mean", "motion_max", "motion_std", "brightness", "sharpness"]
TEXT_STAT_KEYS = ["txt_len", "txt_words", "hashtags", "mentions", "exclaims", "questions",
                  "caps_ratio", "emojis", "has_url", "clickbait"]
CROSS_KEYS = ["clip_sim_mean", "clip_sim_max", "clip_sim_min", "clip_sim_std",
              "temporal_emb_change_mean", "temporal_emb_change_max", "frame_diversity"]


def _as_tensor(out):
    """transformers v4 returns a tensor, v5 may return a model output object."""
    if isinstance(out, torch.Tensor):
        return out
    for k in ("image_embeds", "text_embeds", "pooler_output"):
        v = getattr(out, k, None)
        if v is not None:
            return v
    return out[0]


@torch.no_grad()
def extract(video_path, caption, user_verified=0):
    """Return dict with arrays: frame_emb [8,512], text_emb [512], sbert [384], scalars dict, frames."""
    m = load_models()
    dev = _device()
    frames, vstats = read_video(video_path)
    cap = clean_caption(caption)
    inp = m["proc"](text=[cap], images=frames, return_tensors="pt", padding=True, truncation=True, max_length=77)
    inp = {k: v.to(dev) for k, v in inp.items()}
    img = _as_tensor(m["clip"].get_image_features(pixel_values=inp["pixel_values"]))
    txt = _as_tensor(m["clip"].get_text_features(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"]))
    img = torch.nn.functional.normalize(img, dim=-1).cpu().numpy()
    txt = torch.nn.functional.normalize(txt, dim=-1).cpu().numpy()[0]
    sims = img @ txt
    steps = 1 - np.sum(img[1:] * img[:-1], axis=1)
    centroid = img.mean(0)
    sb = m["sbert"].encode([cap], normalize_embeddings=True)[0]
    scal = {}
    scal.update(vstats)
    scal.update(text_stats(caption))
    scal.update({
        "clip_sim_mean": float(sims.mean()), "clip_sim_max": float(sims.max()),
        "clip_sim_min": float(sims.min()), "clip_sim_std": float(sims.std()),
        "temporal_emb_change_mean": float(steps.mean()), "temporal_emb_change_max": float(steps.max()),
        "frame_diversity": float(1 - np.linalg.norm(centroid)),
    })
    scal["user_verified"] = float(user_verified)
    return {"frame_emb": img.astype(np.float32), "text_emb": txt.astype(np.float32),
            "sbert": sb.astype(np.float32), "scalars": scal, "frames": frames,
            "frame_sims": sims.astype(np.float32)}


SCALAR_KEYS = VIDEO_STAT_KEYS + TEXT_STAT_KEYS + CROSS_KEYS + ["user_verified"]


def tabular_vector(frame_emb, text_emb, sbert, scalars):
    """Flat vector for tree / linear models."""
    s = np.array([scalars[k] for k in SCALAR_KEYS], dtype=np.float32)
    return np.concatenate([frame_emb.mean(0), frame_emb.std(0), text_emb, sbert,
                           frame_emb.mean(0) * text_emb, s]).astype(np.float32)


def tabular_names():
    n = []
    n += [f"vid_mean_{i}" for i in range(512)]
    n += [f"vid_std_{i}" for i in range(512)]
    n += [f"clip_txt_{i}" for i in range(512)]
    n += [f"sbert_{i}" for i in range(384)]
    n += [f"vt_inter_{i}" for i in range(512)]
    n += SCALAR_KEYS
    return n


def feature_group(name):
    if name.startswith(("vid_mean", "vid_std")):
        return "Visual content (CLIP frames)"
    if name.startswith(("clip_txt", "sbert")):
        return "Caption meaning (text)"
    if name.startswith("vt_inter") or name in CROSS_KEYS:
        return "Text-video consistency"
    if name in VIDEO_STAT_KEYS:
        return "Editing & video stats"
    if name in TEXT_STAT_KEYS:
        return "Caption style"
    return "Publisher"
