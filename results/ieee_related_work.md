# VeriFrame: IEEE related work for the comparison table (checked 2026-10-09)

Every DOI below was checked against the Crossref API (prefix 10.1109). Titles, authors and abstracts were checked against the Semantic Scholar Graph API (`/paper/DOI:<doi>`). A metric counts as "verified" only if it appears in the abstract or in an arXiv full text I opened. Links have the form https://doi.org/<DOI>.

## Tier 1: IEEE papers that test on FakeTT / FakeSV (closest comparisons)

| # | Paper | Venue | DOI | Dataset(s) | Method | Reported metric | Comparable to FakeTT? |
|---|---|---|---|---|---|---|---|
| 1 | Li, S. et al., "Multimodal Learning for Fake News Detection in Short Videos Using Linguistically Verified Data and Heterogeneous Modality Fusion" (HFN) | IEEE GLOBECOM 2025 | 10.1109/globecom59602.2025.11432745 (arXiv 2509.15578) | FakeTT, VESV (new, 603 TikTok videos) | Video Swin + CLAP audio + text; DecisionNet modality weighting + weighted attention fusion + LSTM | **FakeTT: Acc 81.73, Macro-F1 79.85.** Video-only Macro-F1 75.47. The same table quotes FakingRecipe at 79.15 / 77.74 and SVFEND at 77.14 / 75.63. Protocol: random 6-fold split (1328/332/332), averaged over 3 runs. | **Yes, directly.** The split protocol is different (random, not temporal). |
| 2 | Huang, X.-J. et al., "Knowledge-Enhanced Dynamic Scene Graph Attention Network for Fake News Video Detection" (KDSGAT) | IEEE Trans. Multimedia, 2026 (DOI issued 2025) | 10.1109/tmm.2025.3623491 | FakeSV, FakeTT | BERT text + HuBERT audio + Swin visual; dynamic scene graphs over keyframes; external knowledge-graph distillation; co-attention fusion | Abstract gives only "+1.86% / +2.68% accuracy over SOTA" on the two datasets. Absolute numbers not verified. | Yes (FakeTT), but absolute numbers not verified |
| 3 | Kong, X. et al., "Harmony in Chaos: A Progressive Noise-Resilient Network for Robust Fake News Video Detection" (PNRN) | IEEE ICME 2025 | 10.1109/icme59968.2025.11208997 | Fake-news-video benchmarks; abstract doesn't name them (likely FakeSV/FakeTT, not verified) | Information-bottleneck unimodal denoising + mixture-of-experts adaptive fusion | not verified | Probably |
| 4 | Li, Y.-L. et al., "REAL: Retrieval-Augmented Prototype Alignment for Improved Fake News Video Detection" | IEEE ICME 2025 | 10.1109/icme59968.2025.11209008 | "three benchmarks" (names not verified; code at github.com/Jian-Lang/REAL) | LLM-driven video retriever + dual (real/fake) prototype alignment, plug-in to existing detectors | not verified | Probably |
| 5 | Zhang, Y. et al., "Confidence Breeds Success: Improving Fake News Video Detection via LVLM-Assisted Inference" (IFAI) | IEEE ICME 2025 | 10.1109/icme59968.2025.11209223 | "benchmark datasets" (not named in abstract) | Large vision-language model produces auxiliary semantics; a key-information selection module feeds a small model | not verified | Probably |
| 6 | Deng, X. et al., "Denoising-Enhanced Multimodal Detection Network for Fake News Video Detection" (D²et) | IEEE Trans. Circuits Syst. Video Technol. (TCSVT), 2026 | 10.1109/tcsvt.2026.3672169 | "two datasets" (not named in abstract; likely FakeSV/FakeTT) | LLM-mined global/local tags guide multimodal feature denoising | not verified | Probably |
| 7 | Ren, S. et al., "MMSFD: Multi-grained and Multi-modal Fusion for Short Video Fake News Detection" | IEEE DSIT 2024 (7th Int. Conf. Data Science & IT) | 10.1109/dsit61374.2024.10881540 | "a real-world short video dataset" (likely FakeSV, not verified) | Fine- and coarse-grained text/visual encoders; similarity-weighted fusion branch + cross-modal attention | not verified | Partly (probably FakeSV, Chinese) |

## Tier 2: IEEE papers on misinformation in short videos / video fake news

| # | Paper | Venue | DOI | Dataset(s) | Method | Reported metric | Comparable? |
|---|---|---|---|---|---|---|---|
| 8 | Shang, L. et al., "A Multimodal Misinformation Detector for COVID-19 Short Videos on TikTok" (TikTec) | IEEE BigData 2021 | 10.1109/bigdata52589.2021.9671928 | Own COVID-19 TikTok dataset | Caption-guided visual + audio (speech) fusion | Own paper's metrics not verified. HFN (#1) reports TikTec reimplemented on FakeTT at Acc 66.22 / Macro-F1 65.08. | Indirectly (via the HFN baseline) |
| 9 | Shang, L. et al., "MultiTec: A Data-Driven Multimodal Short Video Detection Framework for Healthcare Misinformation on TikTok" | IEEE Trans. Big Data, 2025 | 10.1109/tbdata.2025.3533919 | Two TikTok healthcare video datasets | Modality-aware dual-attentive visual/audio/text model | not verified | No (different dataset) |
| 10 | Choi, H. & Ko, Y., "Using Adversarial Learning and Biterm Topic Model for an Effective Fake News Video Detection System on Heterogeneous Topics and Short Texts" | IEEE Access, 2021 | 10.1109/access.2021.3122978 | YouTube fake news videos (title/description + comments) | Biterm topic model + adversarial topic-agnostic network | Abstract: "+3.41%p F1 over previous models". Absolute numbers not verified. (Its CIKM'21 sibling, FANVM, gets FakeTT Acc 71.57 / Macro-F1 70.21 as an HFN baseline.) | No |

## Tier 3: IEEE papers on CLIP-based multimodal fake news (image + text, fallback)

| # | Paper | Venue | DOI | Dataset(s) | Method | Reported metric | Comparable? |
|---|---|---|---|---|---|---|---|
| 11 | Zhou, Y. et al., "Multimodal Fake News Detection via CLIP-Guided Learning" (FND-CLIP) | IEEE ICME 2023 | 10.1109/icme55011.2023.00480 (arXiv 2205.14304) | Weibo, Politifact, GossipCop | BERT + ResNet + CLIP text/image; CLIP-similarity-weighted fusion + modality attention | Acc (arXiv v1): Weibo 0.907, Politifact 0.942, GossipCop 0.880. Not checked against the camera-ready ICME version. | No (image-text posts) |
| 12 | Srivastava, A. et al., "Robust CLIP-Based Multimodal Fake News Detection in Noisy Social Media Environments" | IEEE ICONAT 2025 (Indian conference) | 10.1109/iconat66879.2025.11362532 | Fakeddit (50k subset) | Frozen CLIP image+text embeddings + MLP head, focal loss, modality-aware attention | Abstract: Acc 91.2%, ROC-AUC 97.1%, F1 93% (fake) / 89% (real) | No |

## Venue-level examples (Indian IEEE conferences, ML-style)

| # | Paper | Venue | DOI | Dataset | Method | Reported metric |
|---|---|---|---|---|---|---|
| 13 | Tufchi, S. & Yadav, A., "Advanced Multi-Model Approach for Robust Image-Based Fake News Detection: A Comparative Study of Different Image Models" | IEEE ICCCNT 2024 | 10.1109/icccnt61001.2024.10724156 | IFND (Indian Fake News Dataset) | Swin/ConvNeXt/VGG-19/BEiT/ResNet-50 features + MLP | Abstract: Acc 88.66% (ConvNeXt-Tiny); F1 91% (Swin) |
| 14 | Jannu, O. & Sekar, V., "Comparative Analysis of Deepfake Detection Models" | IEEE I2CT 2024 | 10.1109/i2ct61223.2024.10543823 | Deepfake images (dataset not verified) | Xception, ResNet50, Swin, MobileNet, CNN compared | not verified (abstract is qualitative) |
| 15 | Nandgaonkar, S. S. & Mane, S. B., "Hindi Fake News Detection using Stacking Ensemble Method" | IEEE ICCCNT 2023 | 10.1109/icccnt56998.2023.10307443 | 3 Hindi news datasets | TF-IDF + SVM/LR/DT/NB/RF + stacking ensemble | not verified (abstract is qualitative) |

## Where VeriFrame stands
- **On FakeTT, VeriFrame's numbers (86.1% Acc / 83.4% Macro-F1 / 0.932 AUC) are higher than the best verified IEEE result: HFN at 81.73 / 79.85.** HFN's table also has FakingRecipe at 79.15 / 77.74. VeriFrame's video-only score (79.5–80.1% Acc) is also above HFN's video-only result (Macro-F1 75.47). But these numbers can't be compared one-to-one. VeriFrame tests on a 166-video temporal split using 1,299 of 1,992 videos. HFN averages 3 runs of a random 6-fold split over the full set. The paper should say this, or VeriFrame should be re-run on the FakingRecipe/HFN protocol.
- VeriFrame is the only system in this set that relies on frozen CLIP features plus classic ML ensembles. The others are end-to-end deep fusion networks (KDSGAT, PNRN, D²et, REAL, IFAI), and several use LLMs or LVLMs. That makes VeriFrame cheap and easy to reproduce, which is a fair framing, not a weakness.
- In method, VeriFrame's closest peers are FND-CLIP and Srivastava et al. (CLIP fusion). In venue level, its closest peers are ICCCNT/I2CT/ICONAT papers. Those mostly test on text or image datasets and report single splits.

## Gaps
- Absolute FakeTT/FakeSV numbers for KDSGAT, PNRN, REAL, IFAI, D²et and MMSFD are not verified. Their abstracts give no figures, and there was no open full text (IEEE Xplore needs JS/login). Get them from the PDFs before putting them in the table.
- I found no IEEE paper on FVC (FakeVideoCorpus).
- I found no IEEE INCET/ICACCS paper on fake *video* detection with ML.

## Update 2026-10-09: two older papers replaced
Removed: Choi & Ko, IEEE Access 2021 (10.1109/access.2021.3122978) and Nandgaonkar & Mane, ICCCNT 2023 (10.1109/icccnt56998.2023.10307443).
Added (both verified in Crossref + Semantic Scholar):
- Li, N., Wang, W., Cao, X. "Multi-Domain Short Video Anomaly News Detection." IEEE ICASSP 2026. DOI 10.1109/icassp55912.2026.11463594. Short-video fake/anomalous news across domains. Abstract closed; dataset/metrics not verified (likely FakeSV/FakeTT, check the PDF).
- Antony Seba P. et al. "AI-Powered Sentiment Analysis and Misinformation Detection in Video News Content." 2025 IEEE 9th Int. Conf. on Information and Communication Technology (CICT). DOI 10.1109/cict67193.2025.11399113. NLP on video-news transcripts (multilingual, sentiment + misinformation); Indian-author venue-level example. Metrics not verified.
