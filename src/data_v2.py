"""Loads data/features.npz + data/audio_features.npz into aligned arrays for VeriFrame v2."""
from pathlib import Path
import numpy as np
from .features import SCALAR_KEYS
from .audio_features import consistency
from .v2 import SCALARS_V2, TEXT_IDX, CAPMISS_IDX

ROOT = Path(__file__).resolve().parents[1]


def load_all():
    D = np.load(ROOT / "data/features.npz", allow_pickle=True)
    A = np.load(ROOT / "data/audio_features.npz", allow_pickle=True)
    assert (D["vid"] == A["vid"]).all(), "features.npz and audio_features.npz are not aligned"
    n = len(D["vid"])
    has_sp = A["audio_scalars"][:, 6] > 0.5          # has_speech
    scal, fs_sp = [], []
    for i in range(n):
        c, sf = consistency(D["frame_emb"][i], D["text_emb"][i], D["sbert"][i], A["cap_clap"][i],
                            A["sp_clip"][i], A["sp_sbert"][i], A["clap_audio"][i], True, bool(has_sp[i]))
        scal.append(np.concatenate([D["scalars"][i], A["audio_scalars"][i], [c[k] for k in
                    ["cap_speech_sim", "speech_frame_sim_mean", "speech_frame_sim_max", "speech_frame_sim_min",
                     "cap_audio_sim", "cap_missing", "speech_missing"]]]))
        fs_sp.append(sf)
    present = np.stack([np.ones(n), has_sp.astype(float), A["audio_scalars"][:, 0], A["has_bio"]], 1).astype(np.float32)
    return {
        "vid": D["vid"], "y": D["label"].astype(int), "split": D["split"],
        "frames": D["frame_emb"].astype(np.float32), "fs_cap": D["frame_sims"].astype(np.float32),
        "fs_sp": np.stack(fs_sp).astype(np.float32),
        "text_emb": D["text_emb"], "sbert": D["sbert"],
        "cap": np.concatenate([D["text_emb"], D["sbert"]], 1).astype(np.float32),
        "sp": np.concatenate([A["sp_clip"], A["sp_sbert"]], 1).astype(np.float32),
        "sp_clip": A["sp_clip"], "sp_sbert": A["sp_sbert"],
        "aud": A["clap_audio"].astype(np.float32), "bio": A["bio_sbert"].astype(np.float32),
        "scal": np.stack(scal).astype(np.float32), "present": present,
        "transcript": A["transcript"], "event": A["event"], "domain": A["domain"],
    }


def video_only_view(scal_norm, fs_cap, present):
    """Strip every caption / uploader-derived input (what the dashboard does when no caption is given)."""
    s = scal_norm.copy(); s[..., TEXT_IDX] = 0; s[..., CAPMISS_IDX] = 1
    p = present.copy(); p[..., 0] = 0; p[..., 3] = 0
    return s, np.zeros_like(fs_cap), p
