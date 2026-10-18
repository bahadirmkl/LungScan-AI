# LungScan-AI

**Yapay zekâ tabanlı akciğer kanseri tespiti ve görüntülenmesi (LungScan)**

Akciğer BT (toraks bilgisayarlı tomografi) kesitlerinde kanserli lezyonun otomatik tespiti ve segmentasyonu.

Bu proje, DICOM formatındaki akciğer BT serilerini alıp her kesiti *normal* / *anormal* olarak
sınıflandıran, anormal kesitlerdeki lezyon bölgesini segmente ederek işaretleyen ve sonucu
tekrar DICOM formatında üreten uçtan uca bir çalışma hattıdır.

> Bitirme projesi olarak geliştirilmiştir. Kod, geliştirme sürecindeki farklı yaklaşımların
> (CNN sınıflandırma, SVM, U-Net, Mask R-CNN, YOLOv8-seg) karşılaştırılmasını ve nihai
> uygulamayı bir arada içerir.

---

## İçindekiler

- [Genel Bakış](#genel-bakış)
- [Sistem Mimarisi](#sistem-mimarisi)
- [Depo Yapısı](#depo-yapısı)
- [Kurulum](#kurulum)
- [Model Ağırlıkları](#model-ağırlıkları)
- [Kullanım](#kullanım)
- [Veri Kümesi](#veri-kümesi)
- [Sonuçlar](#sonuçlar)
  - [Örnek Çıktılar](#örnek-çıktılar)
- [Notebook Rehberi](#notebook-rehberi)
- [Gizlilik ve Etik Not](#gizlilik-ve-etik-not)
- [Lisans](#lisans)

---

## Genel Bakış

Geleneksel olarak akciğer BT değerlendirmesi radyologların dilim dilim incelemesini gerektirir ve
zaman alıcıdır. LungScan, bu süreci iki aşamalı bir derin öğrenme hattıyla otomatikleştirmeyi amaçlar:

| Aşama | Görev | Model |
|---|---|---|
| 1 | Kesit sınıflandırma (normal / anormal) | Keras CNN (`model_in_5_1.h5`) |
| 2 | Lezyon segmentasyonu | YOLOv8m-seg (`best.pt`) |

Sınıflandırma aşaması, incelenecek kesit sayısını ciddi biçimde azaltır; segmentasyon aşaması ise
yalnızca anormal bulunan kesitlerde çalışarak lezyonun sınırını görünür hale getirir.

## Sistem Mimarisi

Projenin özgün akış şeması — hasta verisinin PACS'ten indirilmesinden sonuçların
görüntülenmesine kadar tüm süreç:

![LungScan akış şeması](docs/images/akis-semasi.png)

Bileşen düzeyindeki akış:

```
  ┌─────────────────┐
  │  DICOM serisi   │   kaynak_klasor/
  └────────┬────────┘
           │  (1) InstanceNumber'a göre sıralama, aralık filtresi
           ▼
  ┌─────────────────┐
  │   PNG kesitler  │   hedef_klas/
  └────────┬────────┘
           │  (2) Keras CNN sınıflandırıcı
           ▼
   ┌───────┴────────┐
   ▼                ▼
 normal/         abnormal/
   │                │  (3) YOLOv8m-seg lezyon segmentasyonu
   │                ▼
   │           segmented/         (lezyon konturu çizilmiş kesitler)
   │                │
   └───────┬────────┘
           │  (4) Birleştirme
           ▼
        output/
           │  (5) Orijinal DICOM meta verisiyle eşleştirme
           ▼
   hasta_sonuclari/  ──(6)──►  Dahili DICOM görüntüleyici
```

## Depo Yapısı

```
LungScan-AI/
├── app/
│   ├── lungscan_gui.py                  # Nihai Tkinter arayüzü (uçtan uca hat)
│   ├── run_lungscan.bat                 # Windows başlatıcı
│   └── legacy/
│       └── lungscan_gui_v1.py           # Arayüzün önceki sürümü (arşiv)
├── notebooks/
│   ├── 01_veri_hazirlama.ipynb          # DICOM → PNG, etiket eşleştirme
│   ├── 02_siniflandirma_cnn.ipynb       # CNN sınıflandırıcı eğitimi (softmax skorları)
│   ├── 03_siniflandirma_inceptionv3.ipynb
│   ├── 04_siniflandirma_svm.ipynb       # Klasik makine öğrenmesi karşılaştırması
│   ├── 05_konfuzyon_matrisi.ipynb       # Değerlendirme metrikleri
│   ├── 06_segmentasyon_unet.ipynb
│   ├── 07_segmentasyon_maskrcnn_detectron2.ipynb
│   ├── 08_segmentasyon_yolov8.ipynb     # ★ Nihai segmentasyon modelinin eğitimi
│   ├── 09_mask_uretimi.ipynb            # Etiketlerden maske üretimi
│   ├── 10_ara_kodlar.ipynb              # Yardımcı / deneysel kod parçaları
│   └── deneyler/                        # Model boyutu karşılaştırması
│       ├── yolov8n.ipynb
│       ├── yolov8n_50epoch.ipynb
│       ├── yolov8m.ipynb
│       ├── yolov8l.ipynb
│       ├── yolov8x.ipynb
│       └── detectron2_maske_uretimi.ipynb
├── scripts/
│   ├── annotation_filter.py             # Boş segmentasyonlu etiketleri ayıklar
│   ├── json_temizleme.ipynb             # COCO JSON temizleme
│   ├── train_yolov8_seg.py              # Eğitimin betik sürümü
│   ├── qupath_export_json.groovy        # QuPath: görüntü başına COCO JSON
│   └── qupath_merge_json.groovy         # QuPath: tek COCO JSON'da birleştirme
├── configs/
│   ├── lungscan_yolo_dataset.yaml       # YOLO veri kümesi tanımı
│   └── mask_rcnn_R_50_FPN_3x.yaml       # Detectron2 model yapılandırması
├── results/
│   ├── train_yolov8m_seg/               # Eğitim eğrileri ve karmaşıklık matrisi
│   └── val_yolov8m_seg/                 # Doğrulama tahmin görselleri
├── docs/
│   └── images/                          # Akış şeması ve örnek çıktılar
├── requirements.txt
├── CITATION.cff                         # Akademik atıf bilgisi
├── DISCLAIMER.md                        # Yasal ve etik uyarı
└── LICENSE                              # MIT
```

> **Not:** Veri kümesi ve model ağırlıkları, hasta gizliliği ve dosya boyutu nedeniyle
> bu depoya **dahil edilmemiştir**; nasıl elde edileceği aşağıda açıklanmıştır.

## Kurulum

```bash
git clone https://github.com/bahadirmkl/LungScan-AI.git
cd LungScan-AI
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Proje Python 3.10–3.11 ile geliştirilmiştir.

> **Apple Silicon (M serisi) notu:** TensorFlow için
> `pip install tensorflow-macos` gerekebilir. Yalnızca notebook'ları inceleyecekseniz
> TensorFlow kurmak zorunda değilsiniz.

## Model Ağırlıkları

Ağırlıklar depo boyutunu şişirmemek için [**Releases**](https://github.com/bahadirmkl/LungScan-AI/releases)
bölümünde yayınlanır. İndirip `app/` klasörüne koyun:

| Dosya | Açıklama | Boyut |
|---|---|---|
| `best.pt` | YOLOv8m-seg lezyon segmentasyon modeli | ~52 MB |
| `model_in_5_1.h5` | Keras CNN sınıflandırıcı (normal / anormal) — **uygulama bunu kullanır** | ~298 MB |
| `model_in_3.h5` | Keras CNN sınıflandırıcının önceki sürümü (arşiv) | ~298 MB |

```bash
# v1.0.0 sürümündeki ağırlıkları indir
BASE=https://github.com/bahadirmkl/LungScan-AI/releases/download/v1.0.0

curl -L -o app/best.pt          "$BASE/best.pt"
curl -L -o app/model_in_5_1.h5 "$BASE/model_in_5_1.h5"
```

Doğrulama (indirdikten sonra):

```bash
shasum -a 256 app/best.pt
# 9ef3dfbce1694af43a0f07f7152211904209536750f27ea8e0da8304b22742d2
```

Ağırlıkları sıfırdan üretmek isterseniz `notebooks/02_siniflandirma_cnn.ipynb` (sınıflandırıcı)
ve `notebooks/08_segmentasyon_yolov8.ipynb` (segmentasyon) notebook'larını kullanabilirsiniz.

## Kullanım

```bash
cd app
python lungscan_gui.py
```

Arayüz altı adımı sırayla yürütür:

1. **Start Process** – DICOM serisini temizler, verilen indeks aralığındaki kesitleri PNG'ye çevirir.
2. **Start Prediction** – Kesitleri sınıflandırır, anormal olanları segmente eder,
   sonuçları birleştirir ve tekrar DICOM'a dönüştürür.
3. **View Result** – Üretilen DICOM serisini dahili görüntüleyicide açar
   (fare tekerleğiyle dilimler arasında gezinilir).
4. **Restart** – Ara klasörleri temizleyerek başlangıç durumuna döner.

### Çalışma klasörü

Varsayılan olarak betiğin bulunduğu klasör kullanılır. Farklı bir konum için:

```bash
export LUNGSCAN_WORK_DIR=/path/to/calisma_alani
export LUNGSCAN_MODEL_DIR=/path/to/modeller
python app/lungscan_gui.py
```

Beklenen klasör yapısı:

```
<WORK_DIR>/
├── kaynak_klasor/      # ham DICOM serisi (girdi)
├── hedef_klas/         # DICOM → PNG çıktısı
├── normal/             # sınıflandırıcı: normal kesitler
├── abnormal/           # sınıflandırıcı: anormal kesitler
├── segmented/          # lezyon konturu çizilmiş kesitler
├── output/             # normal + segmented birleşimi
└── hasta_sonuclari/    # sonuç DICOM serisi (çıktı)
```

## Veri Kümesi

Eğitim verisi, kurumun PACS sisteminden **Sectra IDS7** ile dışa aktarılan anonimleştirilmiş
toraks BT serilerinden oluşturulmuştur.

| Özellik | Değer |
|---|---|
| Toplam görüntü | 789 eğitim / 198 doğrulama kesiti |
| Sınıf sayısı | 1 (`Cancer`) |
| Etiketleme aracı | QuPath (COCO JSON) → YOLO polygon formatı |
| Görüntü formatı | 8-bit PNG (DICOM'dan dönüştürülmüş) |

Etiketleme akışı:

```
QuPath projesi  ──(qupath_export_json.groovy)──►  görüntü başına COCO JSON
                 ──(qupath_merge_json.groovy)──►  tek COCO JSON
                 ──(annotation_filter.py)──────►  boş segmentasyonlar ayıklanır
                 ──(json_temizleme.ipynb)──────►  YOLO formatı (.txt)
```

Veri kümesi kişisel sağlık verisi içerdiği için bu depoda **paylaşılmamaktadır**.

## Sonuçlar

Nihai model: **YOLOv8m-seg**, 100 epoch, `imgsz=640`, `batch=64`, SGD (`lr0=0.001`),
tek sınıf. Eğitim parametrelerinin tamamı `results/train_yolov8m_seg/args.yaml` içindedir.

| Görev | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| Box (tespit) | 0.894 | 0.807 | 0.893 | 0.576 |
| Mask (segmentasyon) | 0.861 | 0.773 | 0.838 | 0.472 |

![Eğitim eğrileri](results/train_yolov8m_seg/results.png)

Ayrıntılı eğriler, karmaşıklık matrisi ve doğrulama tahminleri için
[`results/`](results/) klasörüne bakın.

Proje sunumunda raporlanan uçtan uca başarı oranları: sınıflandırma modeli **%99.8**,
lezyon tespiti **%99.0**. Bu değerler projeye özel veri kümesi üzerinde elde edilmiştir
ve genelleme performansı hakkında bilgi vermez.

### Örnek Çıktılar

Sınıflandırma + segmentasyon hattının anonimleştirilmiş BT kesitleri üzerindeki
çıktıları (yeşil kutu: tespit edilen lezyon ve güven skoru):

| | |
|---|---|
| ![Örnek 1](docs/images/ornek-tespit-1.png) | ![Örnek 2](docs/images/ornek-tespit-2.png) |
| ![Örnek 3](docs/images/ornek-tespit-3.png) | ![Örnek 4](docs/images/ornek-tespit-4.png) |

## Notebook Rehberi

Proje boyunca farklı yaklaşımlar denenmiş ve karşılaştırılmıştır:

| Yaklaşım | Notebook | Not |
|---|---|---|
| Keras CNN | `02_siniflandirma_cnn.ipynb` | Nihai sınıflandırıcının temeli |
| InceptionV3 | `03_siniflandirma_inceptionv3.ipynb` | Transfer öğrenme denemesi |
| SVM | `04_siniflandirma_svm.ipynb` | Klasik ML referans noktası |
| U-Net | `06_segmentasyon_unet.ipynb` | Piksel bazlı segmentasyon |
| Mask R-CNN | `07_segmentasyon_maskrcnn_detectron2.ipynb` | Detectron2 tabanlı |
| YOLOv8-seg | `08_segmentasyon_yolov8.ipynb` | ★ Nihai model |

Model boyutu etkisi `notebooks/deneyler/` altında n→m→l→x sırasıyla karşılaştırılmıştır.

## Gizlilik ve Etik Not

- Bu depoda **hiçbir hasta verisi, DICOM dosyası veya kişisel sağlık bilgisi bulunmaz.**
- Notebook çıktılarında yer alan örnek kesitler, PACS'ten anonimleştirilerek dışa aktarılmıştır.
- Model ağırlıkları ve veri kümesi ayrı tutulur.
- **Bu yazılım yalnızca akademik araştırma amaçlıdır. Klinik teşhis aracı değildir.**
  Hiçbir tıbbi karar bu sistemin çıktısına dayandırılmamalıdır.

Ayrıntılı yasal ve etik uyarı için: [`DISCLAIMER.md`](DISCLAIMER.md)

## Lisans

[MIT](LICENSE) © 2024 İhsan Bahadır Emekli

Kod MIT lisansı altındadır. Tıbbi kullanım kısıtları için [`DISCLAIMER.md`](DISCLAIMER.md)
dosyasına bakın.

---

## English Summary

**LungScan-AI** is an end-to-end deep learning pipeline for detecting and segmenting
cancerous lesions in thoracic CT scans. It classifies each slice as *normal* or *abnormal*
using a Keras CNN, segments the lesion in abnormal slices with YOLOv8m-seg, and writes the
results back to DICOM format. The repository contains the final application, the full set of
training/evaluation notebooks (CNN, InceptionV3, SVM, U-Net, Mask R-CNN, YOLOv8-seg), and the
data preparation scripts used with QuPath annotations.

Trained YOLOv8m-seg reaches **0.893 box mAP@50** and **0.838 mask mAP@50** on the validation
split. Model weights are distributed via GitHub Releases. For academic research only —
not a clinical diagnostic tool.
