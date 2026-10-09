"""
VeriFrame v2 - Consistency-Guided Tri-Modal Fusion (CGTF).

Tokens: 8 video-frame tokens (CLIP, each tagged with its caption-match and speech-match score),
        caption token (CLIP + MiniLM), speech token (Whisper transcript -> CLIP + MiniLM),
        audio token (CLAP), publisher-bio token (MiniLM), plus a [CLS] token.
A small transformer lets every token attend to every other one; missing modalities are masked out
(key_padding_mask), so ONE network works with any subset of inputs, e.g. video-only (no caption).
Modality dropout during training teaches it to cope with a missing caption / speech / audio / bio.
"""
import numpy as np
import torch
import torch.nn as nn
from .features import SCALAR_KEYS, TEXT_STAT_KEYS
from .audio_features import AUDIO_SCALAR_KEYS, CONSIST_KEYS

SCALARS_V2 = SCALAR_KEYS + AUDIO_SCALAR_KEYS + CONSIST_KEYS
# scalars that come from the caption / uploader metadata -> zeroed in video-only mode
TEXT_DEPENDENT = set(TEXT_STAT_KEYS) | {"clip_sim_mean", "clip_sim_max", "clip_sim_min", "clip_sim_std",
                                        "user_verified", "cap_speech_sim", "cap_audio_sim"}
TEXT_IDX = np.array([i for i, k in enumerate(SCALARS_V2) if k in TEXT_DEPENDENT])
CAPMISS_IDX = SCALARS_V2.index("cap_missing")
MODALITIES = ["caption", "speech", "audio", "bio"]


class CGTF(nn.Module):
    def __init__(self, n_scalars=len(SCALARS_V2), d=192, heads=4, layers=2, drop=0.3, n_frames=8):
        super().__init__()
        def P(i):
            return nn.Sequential(nn.Linear(i, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(drop))
        self.frame = P(512 + 2); self.cap = P(896); self.sp = P(896); self.aud = P(512); self.bio = P(384)
        self.pos = nn.Parameter(torch.zeros(1, n_frames, d))
        self.mtype = nn.Parameter(torch.zeros(1, 6, d))            # cls, frame, cap, speech, audio, bio
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        enc = nn.TransformerEncoderLayer(d, heads, d * 2, drop, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.scal = nn.Sequential(nn.Linear(n_scalars, 64), nn.GELU(), nn.Dropout(drop))
        self.head = nn.Sequential(nn.Linear(d * 2 + 64, 128), nn.GELU(), nn.Dropout(drop), nn.Linear(128, 1))
        nn.init.normal_(self.pos, std=0.02); nn.init.normal_(self.mtype, std=0.02); nn.init.normal_(self.cls, std=0.02)

    def forward(self, frames, fs_cap, fs_sp, cap, sp, aud, bio, scal, present):
        """present: [B,4] float, 1 = modality available (caption, speech, audio, bio)."""
        B, F = frames.shape[:2]
        v = self.frame(torch.cat([frames, fs_cap[..., None], fs_sp[..., None]], -1)) + self.pos + self.mtype[:, 1:2]
        toks = [self.cls.expand(B, -1, -1) + self.mtype[:, 0:1], v,
                self.cap(cap)[:, None] + self.mtype[:, 2:3], self.sp(sp)[:, None] + self.mtype[:, 3:4],
                self.aud(aud)[:, None] + self.mtype[:, 4:5], self.bio(bio)[:, None] + self.mtype[:, 5:6]]
        x = torch.cat(toks, 1)
        pad = torch.cat([torch.zeros(B, 1 + F, dtype=torch.bool, device=x.device), present < 0.5], 1)
        h = self.enc(x, src_key_padding_mask=pad)
        frames_h = h[:, 1:1 + F].mean(1)
        z = torch.cat([h[:, 0], frames_h, self.scal(scal)], -1)
        return self.head(z).squeeze(-1)


# ------------------------------------------------------------------ tabular feature vectors
def tab_full(frame_emb, text_emb, sbert, sp_sbert, clap_audio, bio_sbert, scal):
    return np.concatenate([frame_emb.mean(0), frame_emb.std(0), text_emb, sbert, frame_emb.mean(0) * text_emb,
                           sp_sbert, clap_audio, bio_sbert, scal]).astype(np.float32)


def tab_video(frame_emb, sp_sbert, sp_clip, clap_audio, scal):
    s = scal.copy(); s[TEXT_IDX] = 0
    return np.concatenate([frame_emb.mean(0), frame_emb.std(0), sp_sbert, sp_clip, frame_emb.mean(0) * sp_clip,
                           clap_audio, s]).astype(np.float32)


def tab_full_groups():
    g = (["Visual (CLIP frames)"] * 1024 + ["Caption (text)"] * 896 + ["Caption-video consistency"] * 512 +
         ["Speech (transcript)"] * 384 + ["Audio (CLAP)"] * 512 + ["Publisher bio"] * 384)
    for k in SCALARS_V2:
        if k in CONSIST_KEYS or k.startswith("clip_sim") or k.startswith("temporal") or k == "frame_diversity":
            g.append("Cross-modal consistency")
        elif k in AUDIO_SCALAR_KEYS:
            g.append("Audio (CLAP)")
        elif k in TEXT_STAT_KEYS:
            g.append("Caption style")
        elif k == "user_verified":
            g.append("Publisher bio")
        else:
            g.append("Editing & video stats")
    return g


def tab_video_groups():
    g = (["Visual (CLIP frames)"] * 1024 + ["Speech (transcript)"] * 896 + ["Speech-video consistency"] * 512 +
         ["Audio (CLAP)"] * 512)
    full = tab_full_groups()[-len(SCALARS_V2):]
    return g + full


# columns of tab_full that reproduce the v1 feature set (frames, caption, caption*frame, v1 scalars)
_V1_N_SCAL = len(SCALAR_KEYS)
V1_COLS = np.r_[0:2432, 3712:3712 + _V1_N_SCAL]


def select_v1_cols(X):
    """Used inside sklearn pipelines so v2 can keep the proven v1 feature view as ensemble members."""
    return np.asarray(X)[:, V1_COLS]
