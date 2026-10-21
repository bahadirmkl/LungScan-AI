#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LungScan için sentetik PET/BT benzeri DICOM serisi üretir.

Gerçek hasta verisi paylaşılamadığı için arayüzün **uçtan uca** denenmesini
sağlar: üretilen seri ``kaynak_klasor/`` içine konur, ardından arayüzde
Start Process → Start Prediction → View Result adımları çalıştırılır.

Üretilen görüntüler anatomik olarak **gerçek değildir**; yalnızca akciğer
penceresine benzeyen bir fantomdur. Bu nedenle modelin bu seri üzerindeki
tahminleri klinik bir anlam taşımaz — amaç boru hattının (DICOM okuma,
PNG dönüşümü, sınıflandırma, segmentasyon, DICOM geri yazma, görüntüleme)
hatasız çalıştığını doğrulamaktır.

Kullanım
--------
    python tests/sentetik_dicom_uret.py --output app/kaynak_klasor
    python tests/sentetik_dicom_uret.py --output app/kaynak_klasor --slices 60 --lesion-slices 12

Üretilen dosyalar uzantısızdır (``SYN0001``, ``SYN0002`` ...); gerçek Sectra
dışa aktarımı da bu biçimde dosya adları üretir.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

# Fantom ölçüleri (HU cinsinden doku değerleri).
AIR_HU = -1000.0
SOFT_TISSUE_HU = 40.0
LUNG_HU = -850.0
BONE_HU = 500.0
VESSEL_HU = 60.0
LESION_HU = 120.0

IMAGE_SIZE = 512
PIXEL_SPACING_MM = 0.98
SLICE_THICKNESS_MM = 2.5


def _ellipse(canvas: np.ndarray, cx: float, cy: float, rx: float, ry: float,
             value: float) -> None:
    """Eksen hizalı elipsi ``canvas`` üzerine doldurur (maske ile)."""
    yy, xx = np.ogrid[:canvas.shape[0], :canvas.shape[1]]
    mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    canvas[mask] = value


def _build_slice(index: int, total: int, lesion: bool, rng: np.random.Generator) -> np.ndarray:
    """Tek bir kesitin HU cinsinden piksel dizisini üretir."""
    image = np.full((IMAGE_SIZE, IMAGE_SIZE), AIR_HU, dtype=np.float32)

    # Gövde kesiti: alt/üst uçlarda daralan elips.
    t = index / max(total - 1, 1)
    taper = 1.0 - 0.35 * abs(t - 0.55) / 0.55
    body_rx = 205.0 * taper
    body_ry = 165.0 * taper
    _ellipse(image, 256, 262, body_rx, body_ry, SOFT_TISSUE_HU)

    # Akciğer alanları: merkeze doğru büyüyen iki elips.
    lung_scale = 0.55 + 0.45 * np.sin(np.pi * min(max(t, 0.05), 0.95))
    _ellipse(image, 176, 250, 78 * lung_scale, 112 * lung_scale, LUNG_HU)
    _ellipse(image, 336, 250, 78 * lung_scale, 112 * lung_scale, LUNG_HU)

    # Mediasten ve hilus damarları.
    _ellipse(image, 256, 258, 34, 96, SOFT_TISSUE_HU)
    _ellipse(image, 208, 268, 12, 18, VESSEL_HU)
    _ellipse(image, 304, 268, 12, 18, VESSEL_HU)

    # Omurga ve kaburgalar.
    _ellipse(image, 256, 396, 26, 22, BONE_HU)
    for angle in range(0, 360, 30):
        rad = np.deg2rad(angle)
        _ellipse(image, 256 + 176 * np.cos(rad), 262 + 150 * np.sin(rad), 9, 6, BONE_HU)

    # Sentetik lezyon: sol akciğerde, orta kesitlerde görünen yuvarlak nodül.
    if lesion:
        radius = 16 + 6 * np.sin(np.pi * t)
        _ellipse(image, 188, 238, radius, radius * 0.9, LESION_HU)

    # Gürültü: gerçekçi bir doku dokusu için.
    image += rng.normal(0.0, 18.0, image.shape).astype(np.float32)
    return image


def _write_slice(path: Path, pixels_hu: np.ndarray, index: int, total: int,
                 study_uid: str, series_uid: str, frame_uid: str) -> None:
    """HU dizisini DICOM dosyası olarak yazar."""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = frame_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = CTImageStorage
    dataset.SOPInstanceUID = frame_uid
    dataset.StudyInstanceUID = study_uid
    dataset.SeriesInstanceUID = series_uid
    dataset.FrameOfReferenceUID = generate_uid()

    dataset.PatientName = "SYNTHETIC^PHANTOM"
    dataset.PatientID = "SYNTH-0001"
    dataset.PatientBirthDate = ""
    dataset.PatientSex = "O"
    dataset.StudyID = "1"
    dataset.AccessionNumber = ""
    dataset.StudyDescription = "LungScan sentetik test serisi"
    dataset.SeriesDescription = "Synthetic PET/CT phantom"
    dataset.Modality = "CT"
    dataset.Manufacturer = "LungScan"
    dataset.InstitutionName = "LungScan"

    dataset.SeriesNumber = 1
    dataset.InstanceNumber = index + 1
    dataset.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
    dataset.ImagePositionPatient = [0.0, 0.0, -index * SLICE_THICKNESS_MM]
    dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    dataset.SliceThickness = SLICE_THICKNESS_MM
    dataset.SpacingBetweenSlices = SLICE_THICKNESS_MM
    dataset.PixelSpacing = [PIXEL_SPACING_MM, PIXEL_SPACING_MM]
    dataset.SliceLocation = -index * SLICE_THICKNESS_MM

    dataset.Rows = pixels_hu.shape[0]
    dataset.Columns = pixels_hu.shape[1]
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 1
    dataset.RescaleIntercept = -1024.0
    dataset.RescaleSlope = 1.0
    dataset.RescaleType = "HU"

    # Akciğer penceresi; arayüz VOI LUT uygulayarak görüntüler.
    dataset.WindowCenter = -500.0
    dataset.WindowWidth = 1400.0
    dataset.VOILUTFunction = "LINEAR"

    stored = np.clip(pixels_hu, -1024, 3071).astype(np.int16) + np.int16(1024)
    dataset.PixelData = stored.tobytes()

    dataset.is_little_endian = True
    dataset.is_implicit_VR = False
    dataset.save_as(str(path), write_like_original=False)


def generate(output: Path, slices: int, lesion_slices: int, seed: int,
             clean: bool) -> int:
    """Sentetik seriyi üretir ve yazılan kesit sayısını döndürür."""
    if slices < 1:
        raise ValueError("Kesit sayısı en az 1 olmalıdır.")
    if not 0 <= lesion_slices <= slices:
        raise ValueError("Lezyonlu kesit sayısı 0 ile toplam kesit sayısı arasında olmalıdır.")

    output = Path(output).expanduser().resolve()
    if clean and output.exists():
        for item in output.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    output.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    study_uid = generate_uid()
    series_uid = generate_uid()

    # Lezyonlu kesitler serinin ortasına yerleştirilir.
    first_lesion = max((slices - lesion_slices) // 2, 0)
    last_lesion = first_lesion + lesion_slices

    for index in range(slices):
        pixels = _build_slice(index, slices, first_lesion <= index < last_lesion, rng)
        path = output / f"SYN{index + 1:04d}"
        _write_slice(path, pixels, index, slices, study_uid, series_uid, generate_uid())

    return slices


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="LungScan arayüzü için sentetik DICOM serisi üretir.",
    )
    parser.add_argument("--output", "-o", type=Path, default=Path("app/kaynak_klasor"),
                        help="Serinin yazılacağı klasör (varsayılan: app/kaynak_klasor)")
    parser.add_argument("--slices", "-n", type=int, default=40,
                        help="Üretilecek kesit sayısı (varsayılan: 40)")
    parser.add_argument("--lesion-slices", type=int, default=8,
                        help="Lezyon içerecek kesit sayısı (varsayılan: 8)")
    parser.add_argument("--seed", type=int, default=7,
                        help="Tekrarlanabilirlik için rastgelelik tohumu (varsayılan: 7)")
    parser.add_argument("--clean", action="store_true",
                        help="Yazmadan önce çıktı klasörünü boşalt")
    args = parser.parse_args(argv)

    try:
        written = generate(args.output, args.slices, args.lesion_slices, args.seed, args.clean)
    except Exception as error:                       # noqa: BLE001 - CLI için sade mesaj
        print(f"Hata: {error}", file=sys.stderr)
        return 1

    print(f"{written} sentetik DICOM kesiti yazıldı -> {args.output.resolve()}")
    print("Bu seri yalnızca boru hattı testi içindir; klinik anlamı yoktur.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
