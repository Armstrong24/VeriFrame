# VeriFrame: Multimodal (Text + Video) Fake News Video Detection

Detects fake/misleading short news videos by combining **what the video shows** with **what the caption claims**,
and checking whether the two agree. Trained on **FakeTT** (Bu et al., *FakingRecipe*, ACM MM 2024): English TikTok
videos fact-checked against Snopes.

## Project layout
```
data/fakett_meta.jsonl        FakeTT labels, captions, temporal train/val/test split (1,992 videos)
data/features.npz             pre-extracted features for every video that could still be downloaded
src/features.py               feature extraction (shared by training and the dashboard)
src/fusion_model.py           Cross-Modal Attention Fusion Transformer (PyTorch)
src/predictor.py              loads the trained ensemble, analyzes one video
scripts/01_download_videos.py re-download the TikTok videos (optional)
scripts/02_extract_features.py build data/features.npz from videos (optional)
scripts/03_train.py           train 5 models + tuned ensemble, writes models/ and metrics
models/                       trained models (ready to use)
app/server.py                 FastAPI backend
app/static/                   dashboard (HTML/CSS/JS)
```

## Results (FakeTT temporal test split, 166 unseen videos)
| Model | Accuracy | Macro-F1 | AUC |
|---|---|---|---|
| Logistic Regression | 84.9% | 82.8% | 0.935 |
| SVM (RBF) | 80.7% | 79.0% | 0.939 |
| LightGBM | 80.1% | 78.0% | 0.910 |
| XGBoost | 78.9% | 77.1% | 0.913 |
| Cross-Modal Fusion Transformer | 80.1% | 78.0% | 0.937 |
| **Weighted Ensemble** | **84.3%** | **82.0%** | **0.938** |

Trained on 1,299 FakeTT videos still available on TikTok (train 962 / val 171 / test 166).
Full log: `models/train_log.txt`. Demo videos from the test split are in `demo/`.

## Quick start (VS Code terminal)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.server:app --reload
```
Or just double-click `run_windows.bat` (Windows) / run `./run_mac_linux.sh`.
Open http://127.0.0.1:8000, drop in a video (try the ones in `demo/`), paste its caption and press **Analyze**.
The first run downloads CLIP (~600 MB) and MiniLM (~90 MB) from Hugging Face once.

The models are already trained. To retrain (e.g. after changing hyper-parameters):
```bash
python scripts/03_train.py          # ~5-10 min on CPU, uses data/features.npz
```
To rebuild everything from raw videos:
```bash
python scripts/01_download_videos.py
python scripts/02_extract_features.py
python scripts/03_train.py
```

## Method
**Video branch:** 8 uniformly sampled frames → CLIP ViT-B/32 image embeddings (mean, std, and the full sequence
for the transformer) + editing statistics: shot cuts, cuts per 10 s, motion, duration, fps, resolution, brightness, sharpness.

**Text branch:** caption → CLIP text embedding + MiniLM-L6 sentence embedding + style features
(length, hashtags, mentions, `!`/`?`, CAPS ratio, emojis, URLs, clickbait words) + publisher-verified flag.

**Cross-modal branch:** CLIP caption↔frame similarity per frame (mean/max/min/std), temporal embedding drift,
frame diversity, and element-wise text×video interaction features.

**Models**
| Model | Why |
|---|---|
| Logistic Regression (PCA-256) | linear baseline |
| SVM, RBF kernel (PCA-256) | strong on dense embeddings |
| LightGBM | gradient boosting, gives SHAP explanations in the dashboard |
| XGBoost | second boosting family for ensemble diversity |
| Cross-Modal Attention Fusion Transformer | temporal transformer over frames + text→video and video→text co-attention, 5 seeds averaged |

**Ensemble:** weighted soft-voting; weights and decision threshold are grid-searched on the validation split
for macro-F1, then all models are refit on train+val for deployment. Results are reported on the **temporal
test split** (newest videos), which is harder and more realistic than a random split.

See `models/metrics.json` (also shown on the dashboard's *Model Card* tab) for the numbers.

## Notes and limitations
- Some FakeTT videos have been deleted from TikTok, so training uses the subset that could still be downloaded.
- Audio/speech transcripts are not used (a natural extension: Whisper transcript as a third modality).
- This is fake *news* video detection (misleading context, edited clips, false claims), not only face-swap deepfakes.

## Citation
```
@inproceedings{fakingrecipe,
  title={FakingRecipe: Detecting Fake News on Short Video Platforms from the Perspective of Creative Process},
  author={Bu, Yuyan and Sheng, Qiang and Cao, Juan and Qi, Peng and Wang, Danding and Li, Jintao},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia}, year={2024}}
```
