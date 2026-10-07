"""Download FakeTT TikTok videos by ID with yt-dlp (some are deleted; that's normal)."""
import json, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
VD = ROOT / "videos"; VD.mkdir(exist_ok=True)
ids = [json.loads(l)["video_id"] for l in open(ROOT / "data/fakett_meta.jsonl", encoding="utf-8")]


def get(vid):
    if (VD / f"{vid}.mp4").exists():
        return True
    r = subprocess.run(["yt-dlp", "-q", "--no-warnings", "-f", "worst[ext=mp4]/worst",
                        "-o", str(VD / "%(id)s.%(ext)s"), f"https://www.tiktok.com/@x/video/{vid}"],
                       capture_output=True, timeout=180)
    return (VD / f"{vid}.mp4").exists()


with ThreadPoolExecutor(8) as ex:
    ok = sum(ex.map(get, ids))
print(f"downloaded {ok}/{len(ids)} videos into {VD}")
