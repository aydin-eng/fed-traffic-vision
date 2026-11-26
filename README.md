# 🚦 Fed-Traffic-Vision: Multi-Source Traffic Vehicle Detection & Cross-Domain Federated Learning
## 🎓 Undergraduate Graduation Thesis Project

**Fed-Traffic-Vision** is a modular deep learning framework that unifies **three complementary public traffic datasets** to address class imbalance and domain shift in real-world traffic surveillance cameras, serving as the foundation for **Flower (`flwr`)**–based **Cross-Domain Federated Learning**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aydin-eng/fed-traffic-vision/blob/main/notebooks/01_centralized_3source_yolo11s.ipynb)
[![YOLO11](https://img.shields.io/badge/Model-YOLO11s-orange.svg)](https://docs.ultralytics.com/)
[![Flower FL](https://img.shields.io/badge/Federated-Flower-blueviolet.svg)](https://flower.ai/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

---

## 🧩 3-Source Data Architecture & Unified 4-Class Schema

All sources are harmonized into a canonical 4-class YOLO schema (`car`, `motorcycle`, `bus`, `truck`), while preserving source, camera sequence, and weather metadata for every frame in `dataset_metadata.csv`:

| Class ID | Class | UA-DETRAC | MIO-TCD Localization | COCO 2017 |
|---:|---|---|---|---|
| **0** | `car` | `car`, `van` | `car`, `work_van`, `pickup_truck` | `car` |
| **1** | `motorcycle` | *(none)* | **`motorcycle`** | **`motorcycle`** |
| **2** | `bus` | `bus` | **`bus`** | **`bus`** |
| **3** | `truck` | `others` | **`single_unit_truck`, `articulated_truck`** | **`truck`** |

---

## 🌐 Two-Phase Project Roadmap

### Phase 1: 3-Source Balanced Centralized Baseline (Ready)
* **UA-DETRAC (`src/data/prepare_detrac.py`):** Video-level zero-leakage stratified splitting, class-aware adaptive frame sampling (dense on `truck` & `bus`), and `<ignored_region>` masking.
* **MIO-TCD Localization (`src/data/prepare_miotcd.py`):** Minority-prioritized extraction (`motorcycle`, `single_unit_truck`, `articulated_truck`) from 137,743 North American fixed highway/intersection CCTV frames.
* **COCO 2017 Targeted Subset (`src/data/prepare_coco_subset.py`):** Downloads only the 241 MB annotation bundle and fetches selected minority-vehicle street images in parallel without downloading the full 18 GB archive.
* **Colab T4 Pipeline (`notebooks/01_centralized_3source_yolo11s.ipynb`):** End-to-end `YOLO11s` (`imgsz=800`) training with per-class evaluation and automated Google Drive backup.

### Phase 2: Cross-Domain Federated Learning (Next Stage)
Each dataset domain naturally models a realistic **Non-IID Federated Learning Client**:
* **Client 1 (`ua_detrac`):** East Asian highway CCTV cameras (4 weather domains: `sunny`, `cloudy`, `night`, `rainy`)
* **Client 2 (`mio_tcd`):** North American highway/intersection CCTV cameras (heavy commercial trucks & motorcycles)
* **Client 3 (`coco2017`):** General street-level / multi-angle traffic cameras

Raw camera images remain local on each client; only model parameter updates are aggregated on the central server via **FedAvg / FedProx**.

---

## 🧪 Comprehensive Traffic Data Augmentation (12-Recipe Engine)

To strengthen underrepresented vehicle classes exclusively within the **`train`** split of the unified 3-source vehicle pool (`8,810` train, `1,535` val, `2,397` test)—without touching validation or test ground truth—[src/data/augment_minority.py](src/data/augment_minority.py) implements **12 traffic-camera-specific augmentation recipes**:

* **Geometric & Camera Tilt:** `hflip` (horizontal mirror), `safe_zoom` (`1.12x` center zoom with bounding-box area retention filtering), `mild_rotate_occ` (`±4.5°` camera tilt with exact corner AABB projection + safe roadside occlusion patches never covering >35% of any vehicle box)
* **Weather & Lighting:** `night_lowlight` (gamma darkening + ISO noise), `sun_glare` (daylight highlight glare), `rain_streaks` (angled synthetic rain), `fog_haze` (atmospheric scattering), `overpass_shadow` (roadway bridge shadow band)
* **Sensor & Compound:** `motion_blur`, `cctv_degrade` (sensor Gaussian noise + JPEG Q=45 compression artifacts), `hsv_shift`, `combo_night_flip`
* **Tiered Class Balancing:** **+2 copies** for frames containing `motorcycle` or `bus`, **+1 copy** for `truck`-only minority frames, **0 copies** for `car`-only frames.

![12-Recipe Traffic Augmentation Preview](docs/augmentation_preview.jpg)

| Class | Raw Train Boxes | **Augmented Train Boxes** | Gain | `car:class` Ratio (Before → After) |
|---|---:|---:|---:|---|
| **`car`** | 33,153 | **83,353** | 2.51× | — |
| **`motorcycle`** | 3,497 | **10,488** | **3.00×** | `9.48:1 → 7.95:1` |
| **`bus`** | 3,184 | **9,534** | **2.99×** | `10.41:1 → 8.74:1` |
| **`truck`** | 6,667 | **14,725** | **2.21×** | `4.97:1 → 5.66:1` |
| **Total Train Images** | **8,810** | **21,908** *(+13,098)* | **2.49×** | `val` (`1,535`) & `test` (`2,397`) untouched |

---

## 📁 Directory Structure

```text
fed-traffic-vision/
├── configs/
│   └── dataset_schema.yaml                 # Canonical 4-class schema & 3-source mappings
├── src/
│   ├── data/
│   │   ├── common.py                       # Unicode I/O, masking, YOLO & metadata helpers
│   │   ├── prepare_detrac.py               # UA-DETRAC adaptive & masked extractor
│   │   ├── prepare_miotcd.py               # MIO-TCD Localization extractor
│   │   ├── prepare_coco_subset.py          # COCO 2017 targeted minority fetcher
│   │   ├── augment_minority.py             # Train-only 12-recipe traffic augmentation engine
│   │   └── build_unified_dataset.py        # 3-source unified dataset orchestrator
│   └── federated/                          # Phase 2: Flower FL client/server modules
├── docs/
│   ├── augmentation_preview.jpg            # Visual 3x4 grid of traffic augmentation recipes
│   └── augmentation_report.json            # Per-class and ratio augmentation report
├── notebooks/
│   └── 01_centralized_3source_yolo11s.ipynb # Colab T4 3-source training notebook
├── test_samples/
│   └── user_test_photo.jpg                 # Independent real-world traffic test image
├── detrac_xml_annotations.zip              # Official UA-DETRAC XML annotation archive
├── requirements.txt
└── README.md
```
