#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QuPath'ten disa aktarilan COCO formatli etiket dosyasini temizler.

QuPath'te bazi annotasyonlar segmentasyon poligonu icermez (bos segmentasyon).
Bu durum YOLO egitiminde gecersiz etiket satirlari olusturur. Bu betik:

  1. Segmentasyonu bos olan annotasyonlari ayiklar.
  2. Ardindan hicbir gecerli annotasyonu kalmayan goruntuleri de listeden cikarir.

Kullanim:
    python scripts/annotation_filter.py \
        --input annotations.json \
        --output cleaned_annotations.json
"""

import argparse
import json
import sys


def filter_annotations(data):
    """Bos segmentasyonlu annotasyonlari ve onlarin goruntulerini ayiklar."""
    filtered_images = []
    filtered_annotations = []

    for image_info in data["images"]:
        annotations_for_image = [
            anno for anno in data["annotations"] if anno["image_id"] == image_info["id"]
        ]

        has_segmentation = False
        for anno in annotations_for_image:
            for segment in anno.get("segmentation", []):
                if segment:                     # segmentasyon alani dolu
                    has_segmentation = True
                    filtered_annotations.append(anno)

        if has_segmentation:
            filtered_images.append(image_info)

    return {
        "info": data.get("info", {}),
        "images": filtered_images,
        "annotations": filtered_annotations,
        "categories": data.get("categories", []),
    }


def main():
    parser = argparse.ArgumentParser(
        description="COCO etiket dosyasindan bos segmentasyonlari temizler",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default="annotations.json",
                        help="QuPath'ten gelen ham COCO JSON")
    parser.add_argument("--output", default="cleaned_annotations.json",
                        help="Temizlenmis COCO JSON")
    args = parser.parse_args()

    try:
        with open(args.input, "r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        sys.exit(f"Girdi dosyasi bulunamadi: {args.input}")
    except json.JSONDecodeError as exc:
        sys.exit(f"Gecersiz JSON ({args.input}): {exc}")

    before_images = len(data.get("images", []))
    before_annotations = len(data.get("annotations", []))

    cleaned = filter_annotations(data)

    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(cleaned, file, indent=4, ensure_ascii=False)

    print(f"Goruntu     : {before_images} -> {len(cleaned['images'])}")
    print(f"Annotasyon  : {before_annotations} -> {len(cleaned['annotations'])}")
    print(f"Kaydedildi  : {args.output}")


if __name__ == "__main__":
    main()
