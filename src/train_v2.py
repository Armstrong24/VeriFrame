"""Training helpers for the CGTF network (used by 05_train_v2.py and 06_experiments.py)."""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from .v2 import CGTF, SCALARS_V2, TEXT_IDX, CAPMISS_IDX
from .audio_features import AUDIO_SCALAR_KEYS, CONSIST_KEYS

SP_IDX = np.array([SCALARS_V2.index(k) for k in ["cap_speech_sim", "speech_frame_sim_mean", "speech_frame_sim_max",
                                                 "speech_frame_sim_min", "n_words", "words_per_sec", "has_speech", "speech_ratio"]])
AUD_IDX = np.array([SCALARS_V2.index(k) for k in AUDIO_SCALAR_KEYS + ["cap_audio_sim"]])
CONS_IDX = np.array([SCALARS_V2.index(k) for k in CONSIST_KEYS[:5] +
                     ["clip_sim_mean", "clip_sim_max", "clip_sim_min", "clip_sim_std"]])


def norm_stats(scal, idx):
    mu, sd = scal[idx].mean(0), scal[idx].std(0)
    sd = np.where(sd < 1e-3, 1.0, sd)
    mu[CAPMISS_IDX], sd[CAPMISS_IDX] = 0.0, 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def apply_mask(s, fsc, fss, p, drop):
    """drop: set of modalities removed for these rows (numpy, in place on copies)."""
    s, fsc, fss, p = s.copy(), fsc.copy(), fss.copy(), p.copy()
    if "caption" in drop:
        s[:, TEXT_IDX] = 0; s[:, CAPMISS_IDX] = 1; fsc[:] = 0; p[:, 0] = 0
    if "bio" in drop:
        p[:, 3] = 0
    if "speech" in drop:
        s[:, SP_IDX] = 0; fss[:] = 0; p[:, 1] = 0
    if "audio" in drop:
        s[:, AUD_IDX] = 0; p[:, 2] = 0
    if "consistency" in drop:
        s[:, CONS_IDX] = 0; fsc[:] = 0; fss[:] = 0
    return s, fsc, fss, p


def tensors(D, idx, mu, sd, drop=frozenset()):
    s = (D["scal"][idx] - mu) / sd
    s, fsc, fss, p = apply_mask(s, D["fs_cap"][idx], D["fs_sp"][idx], D["present"][idx], drop)
    T = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32)
    return [T(D["frames"][idx]), T(fsc), T(fss), T(D["cap"][idx]), T(D["sp"][idx]), T(D["aud"][idx]),
            T(D["bio"][idx]), T(s), T(p)]


def train_cgtf(D, itr, iva=None, seed=0, epochs=40, mdrop=True, ablate=frozenset(), p_cap=0.35, p_other=0.15,
               lr=3e-4, select_on=("full", "video"), patience=7):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    y = D["y"]; mu, sd = norm_stats(D["scal"], itr)
    net = CGTF(); opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    pos_w = (y[itr] == 0).sum() / max((y[itr] == 1).sum(), 1)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, dtype=torch.float32))
    best, best_state, best_ep = -1, None, epochs
    for ep in range(epochs):
        net.train(); perm = rng.permutation(itr)
        for b in range(0, len(perm), 32):
            ix = perm[b:b + 32]
            s = (D["scal"][ix] - mu) / sd
            fsc, fss, p = D["fs_cap"][ix], D["fs_sp"][ix], D["present"][ix]
            s, fsc, fss, p = apply_mask(s, fsc, fss, p, ablate)
            if mdrop:  # modality dropout: hide whole modalities for random rows
                for mod, pr in [("caption", p_cap), ("speech", p_other), ("audio", p_other), ("bio", p_other)]:
                    if mod in ablate:
                        continue
                    rows = rng.random(len(ix)) < pr
                    if rows.any():
                        s2, f2, g2, p2 = apply_mask(s[rows], fsc[rows], fss[rows], p[rows], {mod} | ({"bio"} if mod == "caption" else set()))
                        s[rows], fsc[rows], fss[rows], p[rows] = s2, f2, g2, p2
            T = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32)
            fr = T(D["frames"][ix]); fr = fr + 0.02 * torch.randn_like(fr)
            logit = net(fr, T(fsc), T(fss), T(D["cap"][ix]), T(D["sp"][ix]), T(D["aud"][ix]), T(D["bio"][ix]), T(s), T(p))
            loss = lossf(logit, torch.tensor(y[ix], dtype=torch.float32))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        sched.step()
        if iva is not None:
            aucs = []
            if "full" in select_on:
                aucs.append(roc_auc_score(y[iva], predict(net, D, iva, mu, sd, ablate)))
            if "video" in select_on and mdrop:
                aucs.append(roc_auc_score(y[iva], predict(net, D, iva, mu, sd, ablate | {"caption", "bio"})))
            a = float(np.mean(aucs))
            if a > best:
                best, best_ep = a, ep + 1
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            elif patience and ep + 1 - best_ep >= patience:   # early stopping (keeps CPU time down)
                break
    if best_state:
        net.load_state_dict(best_state)
    net.mu, net.sd, net.best_epoch = mu, sd, best_ep
    return net.eval()


@torch.no_grad()
def predict(net, D, idx, mu=None, sd=None, drop=frozenset()):
    mu = net.mu if mu is None else mu; sd = net.sd if sd is None else sd
    net.eval()
    return torch.sigmoid(net(*tensors(D, idx, mu, sd, drop))).numpy()
