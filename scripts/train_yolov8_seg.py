#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8m-seg lezyon segmentasyon modelinin egitimi ve degerlendirilmesi.

Bu betik, `notebooks/08_segmentasyon_yolov8.ipynb` dosyasinin calistirilabilir
surumudur. Colab'a ozgu hucre sihirleri (`!unzip`, `%cd`, `pip install`) kaldirilmis,
tum yollar komut satiri argumanlarina baglanmistir.

Egitim:
    python scripts/train_yolov8_seg.py train \
        --data configs/lungscan_yolo_dataset.yaml \
        --model yolov8m-seg.pt \
        --epochs 100 --batch 64 --imgsz 640

Degerlendirme (IoU tabanli karmasiklik matrisi dahil):
    python scripts/train_yolov8_seg.py eval \
        --weights runs/train/exp/weights/best.pt \
        --data configs/lungscan_yolo_dataset.yaml \
        --images all_info/images/val \
        --labels all_info/labels/val
"""

import argparse
import os
import sys

import numpy as np

try:
    import cv2
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    sys.exit(f"Eksik bagimlilik: {exc}. Once 'pip install -r requirements.txt' calistirin.")


# ---------------------------------------------------------------------------
# Yardimcilar
# ---------------------------------------------------------------------------
def polygon_labels_to_masks(label_path, height, width):
    """YOLO polygon etiket dosyasini ikili maskelere cevirir."""
    real_masks = []
    if not os.path.exists(label_path):
        return real_masks

    with open(label_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 7:          # class + en az 3 nokta (x,y)
                continue
            _class_label, *polygon = parts
            points = np.array(polygon, dtype=np.float32).reshape(-1, 2)
            mask_image = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask_image, [np.int32(points * [width, height])], 255)
            real_masks.append(mask_image)
    return real_masks


def predicted_masks_to_binary(results, height, width):
    """Model cikti maskelerini orijinal cozunurlukte ikili maskelere cevirir."""
    if results[0].masks is None:
        return []
    masks = []
    for mask in results[0].masks.data.cpu().numpy():
        _, mask_thresh = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
        mask_thresh = mask_thresh.astype(np.uint8)
        if mask_thresh.shape[:2] != (height, width):
            mask_thresh = cv2.resize(mask_thresh, (width, height), interpolation=cv2.INTER_NEAREST)
        masks.append(mask_thresh)
    return masks


def calculate_iou(mask1, mask2):
    """Iki ikili maske arasindaki Intersection over Union skoru."""
    mask1 = np.array(mask1, dtype=np.bool_)
    mask2 = np.array(mask2, dtype=np.bool_)

    if mask1.shape != mask2.shape:
        larger_shape = np.maximum(mask1.shape, mask2.shape)
        mask1 = cv2.resize(mask1.astype(np.uint8), larger_shape[::-1],
                           interpolation=cv2.INTER_NEAREST).astype(np.bool_)
        mask2 = cv2.resize(mask2.astype(np.uint8), larger_shape[::-1],
                           interpolation=cv2.INTER_NEAREST).astype(np.bool_)

    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    if np.sum(union) == 0:
        return 0.0
    return float(np.sum(intersection) / np.sum(union))


def calculate_max_iou(predicted_mask, real_masks):
    """Bir tahmin maskesi icin gercek maskeler arasindaki en yuksek IoU."""
    scores = [calculate_iou(predicted_mask, real_mask) for real_mask in real_masks]
    return max(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Egitim
# ---------------------------------------------------------------------------
def cmd_train(args):
    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        optimizer=args.optimizer,
        project=args.project,
        name=args.name,
        exist_ok=True,
        device=args.device,
        lr0=args.lr0,
    )
    print("\nEgitim tamamlandi.")
    print(f"En iyi agirliklar: {os.path.join(args.project, args.name, 'weights', 'best.pt')}")


# ---------------------------------------------------------------------------
# Degerlendirme
# ---------------------------------------------------------------------------
def cmd_eval(args):
    model = YOLO(args.weights)

    # 1) Ultralytics'in standart dogrulamasi (mAP, precision, recall)
    print("== Ultralytics model.val() ==")
    metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device)
    print(f"  Box  : P={metrics.box.mp:.3f}  R={metrics.box.mr:.3f}  "
          f"mAP50={metrics.box.map50:.3f}  mAP50-95={metrics.box.map:.3f}")
    print(f"  Mask : P={metrics.seg.mp:.3f}  R={metrics.seg.mr:.3f}  "
          f"mAP50={metrics.seg.map50:.3f}  mAP50-95={metrics.seg.map:.3f}")

    if not args.images:
        return

    # 2) IoU tabanli karmasiklik matrisi (tek ornek uzerinde)
    print(f"\n== Ornek gorsel uzerinde IoU analizi ==")
    val_images_dir = args.images
    val_labels_dir = args.labels or args.images

    val_image_files = [f for f in sorted(os.listdir(val_images_dir)) if f.endswith(".png")]
    if not val_image_files:
        print("  Degerlendirilecek PNG bulunamadi.")
        return

    rng = np.random.default_rng(args.seed)
    image_file = args.image if args.image else rng.choice(val_image_files)
    img_path = os.path.join(val_images_dir, image_file)

    img = cv2.imread(img_path)
    if img is None:
        sys.exit(f"Gorsel okunamadi: {img_path}")
    height, width = img.shape[:2]

    results = model(img)
    predicted_masks = predicted_masks_to_binary(results, height, width)

    label_path = os.path.join(val_labels_dir, image_file.replace(".png", ".txt"))
    real_masks = polygon_labels_to_masks(label_path, height, width)

    if not predicted_masks:
        print(f"  {image_file}: tahmin yok.")
        return
    if not real_masks:
        print(f"  {image_file}: gercek maske yok, IoU hesaplanamaz.")
        return

    max_iou_scores = [calculate_max_iou(mask, real_masks) for mask in predicted_masks]

    threshold = args.iou_threshold
    tp = sum(1 for score in max_iou_scores if score >= threshold)
    fp = len(max_iou_scores) - tp
    fn = max(len(real_masks) - tp, 0)

    print(f"  Gorsel          : {image_file}")
    print(f"  IoU esigi       : {threshold}")
    print(f"  Tahmin sayisi   : {len(predicted_masks)}")
    print(f"  Gercek maske    : {len(real_masks)}")
    print(f"  TP={tp}  FP={fp}  FN={fn}")
    print(f"  Ortalama IoU    : {np.mean(max_iou_scores):.3f}")

    for i, score in enumerate(max_iou_scores):
        print(f"    Tahmin {i}: max IoU = {score:.3f}")

    _plot_confusion_matrix(tp, fp, fn, len(real_masks), args.output_dir, image_file)


def _plot_confusion_matrix(tp, fp, fn, total_real, output_dir, image_file):
    """IoU tabanli karmasiklik matrisini cizip kaydeder."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("  (matplotlib/seaborn yok, grafik atlandi)")
        return

    os.makedirs(output_dir, exist_ok=True)
    conf_matrix = np.array([[tp, fp], [fn, max(total_real - tp, 0)]])

    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Predicted Positive", "Predicted Negative"],
                yticklabels=["Actual Positive", "Actual Negative"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"IoU >= {0.5} - {image_file}")
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"confusion_{os.path.splitext(image_file)[0]}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Karmasiklik matrisi kaydedildi: {out_path}")


# ---------------------------------------------------------------------------
# Komut satiri arayuzu
# ---------------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(
        description="LungScan YOLOv8m-seg egitim ve degerlendirme betigi",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="Modeli egit")
    t.add_argument("--data", default="configs/lungscan_yolo_dataset.yaml",
                   help="Veri kumesi tanim dosyasi")
    t.add_argument("--model", default="yolov8m-seg.pt",
                   help="Baslangic agirliklari (onceden egitilmis)")
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--imgsz", type=int, default=640)
    t.add_argument("--optimizer", default="SGD")
    t.add_argument("--lr0", type=float, default=0.001)
    t.add_argument("--project", default="runs/train")
    t.add_argument("--name", default="exp")
    t.add_argument("--device", default="cuda", help="'cuda', 'cpu' veya '0,1'")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("eval", help="Modeli degerlendir")
    e.add_argument("--weights", required=True, help="best.pt yolu")
    e.add_argument("--data", default="configs/lungscan_yolo_dataset.yaml")
    e.add_argument("--imgsz", type=int, default=640)
    e.add_argument("--device", default="cuda")
    e.add_argument("--images", default=None, help="IoU analizi icin val goruntu klasoru")
    e.add_argument("--labels", default=None, help="IoU analizi icin val etiket klasoru")
    e.add_argument("--image", default=None, help="Belirli bir gorsel (varsayilan: rastgele)")
    e.add_argument("--iou-threshold", type=float, default=0.5)
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--output-dir", default="results/eval")
    e.set_defaults(func=cmd_eval)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
