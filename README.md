# 🚦 Fed-Traffic-Vision: Çoklu-Kaynak Trafik Araç Tespiti ve Cross-Domain Federal Öğrenme
## 🎓 Lisans Bitirme Projesi (Graduation Thesis Project)

**Fed-Traffic-Vision**, gerçek dünya trafik gözetleme kameralarında sınıf dengesizliği ve alan kayması (*domain shift*) sorunlarını çözmek için **3 tamamlayıcı açık trafik veri setini** birleştiren ve devamında **Flower (`flwr`)** tabanlı **Cross-Domain Federated Learning** mimarisine temel oluşturan modüler bir derin öğrenme projesidir.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aydin-eng/fed-traffic-vision/blob/main/notebooks/01_centralized_3source_yolo11s.ipynb)
[![YOLO11](https://img.shields.io/badge/Model-YOLO11s-orange.svg)](https://docs.ultralytics.com/)
[![Flower FL](https://img.shields.io/badge/Federated-Flower-blueviolet.svg)](https://flower.ai/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

---

## 🧩 3-Kaynaklı Veri Mimarisi ve Ortak 4-Sınıf Şeması

Tüm kaynaklar tek bir ortak 4-sınıf YOLO şemasında (`car`, `motorcycle`, `bus`, `truck`) birleştirilir ve her karenin kaynak/kamera/hava metaverisi `dataset_metadata.csv` içinde korunur:

| Sınıf ID | Sınıf | UA-DETRAC | MIO-TCD Localization | COCO 2017 |
|---:|---|---|---|---|
| **0** | `car` | `car`, `van` | `car`, `work_van`, `pickup_truck` | `car` |
| **1** | `motorcycle` | *(yok)* | **`motorcycle`** | **`motorcycle`** |
| **2** | `bus` | `bus` | **`bus`** | **`bus`** |
| **3** | `truck` | `others` | **`single_unit_truck`, `articulated_truck`** | **`truck`** |

---

## 🌐 2 Aşamalı Proje Yol Haritası

### Faz 1: 3-Kaynaklı Dengeli Merkezi Baseline (Hazır)
* **UA-DETRAC (`src/data/prepare_detrac.py`):** Video-seviyeli sıfır-sızıntılı (*zero-leakage*) bölme, sınıf-duyarlı adaptive örnekleme (`truck` & `bus` yoğunluklu) ve `<ignored_region>` maskeleme.
* **MIO-TCD Localization (`src/data/prepare_miotcd.py`):** 137.743 karelik Kuzey Amerika sabit CCTV trafik veri setinden `motorcycle` ve ağır vasıta (`single_unit_truck`, `articulated_truck`) öncelikli çıkarım.
* **COCO 2017 Hedefli Alt Küme (`src/data/prepare_coco_subset.py`):** 18 GB arşivin tamamını indirmeden yalnızca azınlık sınıflarını içeren sokak görsellerinin paralel indirilmesi.
* **Colab T4 Pipeline (`notebooks/01_centralized_3source_yolo11s.ipynb`):** Tek tıkla `YOLO11s` (`imgsz=800`) eğitimi ve sınıf bazlı raporlama.

### Faz 2: Cross-Domain Federated Learning (Sıradaki Adım)
Üç veri kaynağı doğrudan gerçekçi **Non-IID İstemciler (Clients)** olarak modellenir:
* **Client 1 (`ua_detrac`):** Doğu Asya otoyol CCTV kameraları (4 hava durumu: `sunny`, `cloudy`, `night`, `rainy`)
* **Client 2 (`mio_tcd`):** Kuzey Amerika kavşak/otoyol CCTV kameraları (yüksek ağır vasıta ve motosiklet yoğunluğu)
* **Client 3 (`coco2017`):** Genel cadde / çoklu kamera açıları

Ham kamera görüntüleri paylaşılmadan yalnızca model ağırlıkları merkezi sunucuda **FedAvg / FedProx** ile birleştirilecektir.

---

## 📁 Dizin Yapısı

```text
fed-traffic-vision/
├── configs/
│   └── dataset_schema.yaml                 # Ortak 4 sınıf ve 3 kaynak eşleme şeması
├── src/
│   ├── data/
│   │   ├── common.py                       # Unicode I/O, maskeleme, YOLO & metadata yardımcıları
│   │   ├── prepare_detrac.py               # UA-DETRAC adaptive & masked extractor
│   │   ├── prepare_miotcd.py               # MIO-TCD Localization extractor
│   │   ├── prepare_coco_subset.py          # COCO 2017 targeted fetcher
│   │   ├── augment_minority.py             # Train-only nadir sınıf augmentation
│   │   └── build_unified_dataset.py        # 3-kaynaklı orkestratör
│   └── federated/                          # Faz 2: Flower FL istemci/sunucu modülleri
├── notebooks/
│   └── 01_centralized_3source_yolo11s.ipynb # Colab T4 3-kaynaklı eğitim notebook'u
├── test_samples/
│   └── user_test_photo.jpg                 # Bağımsız saha test fotoğrafı
├── detrac_xml_annotations.zip              # Resmi UA-DETRAC XML etiket paketi
├── requirements.txt
└── README.md
```
