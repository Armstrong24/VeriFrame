"""Cross-modal attention fusion network (text <-> video frames)."""
import torch
import torch.nn as nn


class CrossModalFusion(nn.Module):
    def __init__(self, n_scalars, d=256, n_frames=8, heads=4, drop=0.3):
        super().__init__()
        self.frame_proj = nn.Sequential(nn.Linear(512, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(drop))
        self.text_proj = nn.Sequential(nn.Linear(512 + 384, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(drop))
        self.pos = nn.Parameter(torch.zeros(1, n_frames, d))
        enc = nn.TransformerEncoderLayer(d, heads, d * 2, drop, batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(enc, num_layers=1)
        self.t2v = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)  # text queries video
        self.v2t = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)  # video queries text
        self.scal = nn.Sequential(nn.Linear(n_scalars, 64), nn.GELU(), nn.Dropout(drop))
        self.head = nn.Sequential(nn.Linear(d * 4 + 64 + 8, 256), nn.GELU(), nn.Dropout(drop), nn.Linear(256, 1))

    def forward(self, frames, clip_txt, sbert, scalars, frame_sims):
        v = self.frame_proj(frames) + self.pos                 # [B,8,d]
        v = self.temporal(v)
        t = self.text_proj(torch.cat([clip_txt, sbert], -1)).unsqueeze(1)  # [B,1,d]
        t2v, attn = self.t2v(t, v, v)                         # [B,1,d]
        v2t, _ = self.v2t(v, t, t)
        z = torch.cat([t.squeeze(1), t2v.squeeze(1), v.mean(1), v2t.mean(1),
                       self.scal(scalars), frame_sims], -1)
        return self.head(z).squeeze(-1), attn.squeeze(1)
