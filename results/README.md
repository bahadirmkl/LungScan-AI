# Sonuçlar

Bu klasör, nihai segmentasyon modelinin (**YOLOv8m-seg**, 100 epoch) eğitim ve doğrulama
çıktılarını içerir.

## `train_yolov8m_seg/`

Eğitim sırasında (her epoch sonunda) üretilen değerlendirme görselleri.

| Dosya | Açıklama |
|---|---|
| `args.yaml` | Eğitim hiperparametrelerinin tam dökümü |
| `results.csv` | Epoch bazında tüm metrikler (loss, precision, recall, mAP) |
| `results.png` | Kayıp ve metrik eğrilerinin özeti |
| `Box*_curve.png` | Tespit (bounding box) için P / R / F1 / PR eğrileri |
| `Mask*_curve.png` | Segmentasyon maskesi için P / R / F1 / PR eğrileri |
| `confusion_matrix.png` | Karmaşıklık matrisi (mutlak) |
| `confusion_matrix_normalized.png` | Karmaşıklık matrisi (normalize edilmiş) |
| `labels.jpg` / `labels_correlogram.jpg` | Veri kümesi etiket dağılımı analizi |
| `val_batch*_labels.jpg` | Doğrulama batch'lerinin gerçek etiketleri |
| `val_batch*_pred.jpg` | Aynı batch'lerin model tahminleri |

## `val_yolov8m_seg/`

Eğitim sonrası bağımsız `model.val()` çalıştırmasının çıktıları (aynı düzen).

## Özet Metrikler

Son epoch (100) itibarıyla:

| Görev | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| Box | 0.894 | 0.807 | 0.893 | 0.576 |
| Mask | 0.861 | 0.773 | 0.838 | 0.472 |

Model 1 sınıf (`Cancer`) üzerinde eğitilmiştir. Eğitim 100 epoch, `imgsz=640`,
`batch=64` ve SGD (`lr0=0.001`, `lrf=0.01`) ile yapılmıştır.

## Metrikleri yeniden üretmek

```bash
yolo segment val \
  model=runs/train/exp/weights/best.pt \
  data=configs/lungscan_yolo_dataset.yaml \
  imgsz=640
```

> **Not:** Eğitim Google Colab üzerinde yürütülmüştür (bkz. `args.yaml` içindeki
> `project` yolu). Grafikleri yeniden üretmek için veri kümesine ihtiyaç vardır;
> veri kümesi hasta gizliliği nedeniyle bu depoda bulunmaz.
