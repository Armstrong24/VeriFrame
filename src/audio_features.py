"""
VeriFrame v2 - Audio + Speech branch (shared by training and the dashboard).

New modalities
  - Speech : Whisper (base.en, faster-whisper int8) transcript of what is SAID in the video
  - Audio  : CLAP (laion/clap-htsat-unfused) embedding of the soundtrack + loudness / speech-ratio stats
  - Profile: publisher self-description (optional, from TikTok bio)

New cross-modal consistency signals (the core idea of v2)
  - caption  <-> speech   (MiniLM cosine)     does the caption claim what the narrator says?
  - speech   <-> frames   (CLIP, per frame)   does the narration match what is shown?   (works with NO caption)
  - caption  <-> audio    (CLAP text-audio)   does the soundtrack fit the claim?
  - audio type scores     (CLAP zero-shot: speech, music, news broadcast, crowd, silence)
"""
import numpy as np
import torch

WHISPER_NAME = "base.en"
CLAP_NAME = "laion/clap-htsat-unfused"
AUDIO_PROMPTS = ["a person speaking", "background music", "a news broadcast", "a crowd cheering", "silence"]
CLAP_WINDOWS = 3          # up to 3 x 10 s windows, embeddings averaged
_m = {}


def load_audio_models():
    if not _m:
        from faster_whisper import WhisperModel
        from transformers import ClapModel, ClapProcessor
        _m["whisper"] = WhisperModel(WHISPER_NAME, device="cpu", compute_type="int8",
                                     cpu_threads=max(1, torch.get_num_threads()))
        _m["clap"] = ClapModel.from_pretrained(CLAP_NAME).eval()
        _m["clap_proc"] = ClapProcessor.from_pretrained(CLAP_NAME)
        with torch.no_grad():
            _m["prompt_emb"] = clap_text(AUDIO_PROMPTS)
    return _m


def _t(out, keys=("audio_embeds", "text_embeds", "pooler_output")):
    if isinstance(out, torch.Tensor):
        return out
    for k in keys:
        v = getattr(out, k, None)
        if v is not None:
            return v
    return out[0]


@torch.no_grad()
def clap_text(texts):
    m = _m
    inp = m["clap_proc"](text=list(texts), return_tensors="pt", padding=True, truncation=True)
    e = _t(m["clap"].get_text_features(**inp))
    return torch.nn.functional.normalize(e, dim=-1).numpy().astype(np.float32)


def _decode(path, sr):
    from faster_whisper.audio import decode_audio
    try:
        return decode_audio(str(path), sampling_rate=sr)
    except Exception:
        return np.zeros(sr, dtype=np.float32)          # video without an audio track


AUDIO_SCALAR_KEYS = ["has_audio", "rms_mean", "rms_std", "speech_ratio", "n_words", "words_per_sec",
                     "has_speech"] + [f"aud_{p.split()[-1]}" for p in AUDIO_PROMPTS]


@torch.no_grad()
def extract_audio(video_path, max_seconds=120):
    """Returns dict: transcript (str), clap_audio [512], audio scalars dict."""
    m = load_audio_models()
    a16 = _decode(video_path, 16000)[: 16000 * max_seconds]
    has_audio = float(np.abs(a16).max() > 1e-4) if len(a16) else 0.0
    dur = max(len(a16) / 16000, 1e-3)
    # ---- speech
    text, speech_s = "", 0.0
    if has_audio:
        segs, _ = m["whisper"].transcribe(a16, beam_size=1, vad_filter=True, language="en", temperature=0.0,
                                          condition_on_previous_text=False)  # no slow temperature-fallback loops
        segs = list(segs)
        text = " ".join(s.text.strip() for s in segs).strip()
        speech_s = sum(s.end - s.start for s in segs)
    n_words = len(text.split())
    # ---- soundtrack (CLAP @ 48 kHz)
    a48 = _decode(video_path, 48000)[: 48000 * max_seconds]
    win = 48000 * 10
    starts = np.linspace(0, max(len(a48) - win, 0), CLAP_WINDOWS).astype(int) if len(a48) > win else [0]
    clips = [a48[s:s + win] for s in sorted(set(starts))]
    inp = m["clap_proc"](audio=clips, sampling_rate=48000, return_tensors="pt")
    ae = torch.nn.functional.normalize(_t(m["clap"].get_audio_features(**inp)), dim=-1).mean(0)
    ae = torch.nn.functional.normalize(ae, dim=-1).numpy().astype(np.float32)
    zs = (m["prompt_emb"] @ ae) * 10
    zs = np.exp(zs - zs.max()); zs = zs / zs.sum()
    rms = np.sqrt(np.convolve(a16 ** 2, np.ones(1600) / 1600, mode="valid")[::800] + 1e-12) if len(a16) > 1600 else np.zeros(1)
    sc = {"has_audio": has_audio, "rms_mean": float(np.log10(rms.mean() + 1e-6)), "rms_std": float(rms.std() / (rms.mean() + 1e-6)),
          "speech_ratio": float(min(speech_s / dur, 1.0)), "n_words": float(np.log1p(n_words)),
          "words_per_sec": float(n_words / dur), "has_speech": float(n_words >= 3)}
    for p, z in zip(AUDIO_PROMPTS, zs):
        sc[f"aud_{p.split()[-1]}"] = float(z)
    return {"transcript": text, "clap_audio": ae, "audio_scalars": sc}


# ------------------------------------------------------------------ consistency features
CONSIST_KEYS = ["cap_speech_sim", "speech_frame_sim_mean", "speech_frame_sim_max", "speech_frame_sim_min",
                "cap_audio_sim", "cap_missing", "speech_missing"]


def consistency(frame_emb, cap_clip, cap_sbert, cap_clap, sp_clip, sp_sbert, clap_audio, has_caption, has_speech):
    """All embeddings are L2-normalised. Missing modalities give 0 and a *_missing flag."""
    sf = frame_emb @ sp_clip if has_speech else np.zeros(len(frame_emb), np.float32)
    return {
        "cap_speech_sim": float(cap_sbert @ sp_sbert) if (has_caption and has_speech) else 0.0,
        "speech_frame_sim_mean": float(sf.mean()), "speech_frame_sim_max": float(sf.max()),
        "speech_frame_sim_min": float(sf.min()),
        "cap_audio_sim": float(cap_clap @ clap_audio) if has_caption else 0.0,
        "cap_missing": float(not has_caption), "speech_missing": float(not has_speech),
    }, sf.astype(np.float32)
