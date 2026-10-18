#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LungScan — Akciğer PET/BT (DICOM) kesitlerinde kanserli lezyon tespiti ve
konturlanması için masaüstü arayüz.

Akış
----
    1) Start Process    : DICOM serisini InstanceNumber'a göre sıralar ve seçilen
                          indeks aralığındaki kesitleri PNG'ye çevirir.
    2) Start Prediction : Kesitleri normal/anormal olarak sınıflandırır, anormal
                          kesitlerde lezyon konturunu çizer, kesitleri birleştirir
                          ve sonucu tekrar DICOM'a çevirir.
    3) View Result      : Üretilen DICOM serisini dahili görüntüleyicide açar.
    4) Restart          : Ara klasörleri temizler, kaynak veriye dokunmaz.

Tasarım ilkeleri
----------------
* **Kaynak klasöre hiçbir zaman yazılmaz veya silinmez.** Girdi DICOM'ları
  ``kaynak_klasor/`` altında salt-okunur kalır; tüm ara ve son ürünler ayrı
  klasörlerde üretilir. Akış bu nedenle tekrar çalıştırılabilir.
* Kesit seçimi ``Start Index`` / ``End Index`` ile yapılır. Aralık dışındaki
  kesitler **silinmez**, yalnızca işlenmez.
* Arayüz iş parçacığı (worker thread) hiçbir Tk çağrısı yapmaz; mesajlar
  ``queue.Queue`` üzerinden ana iş parçacığına aktarılır (Tkinter thread-safe
  değildir).
* Sonuç DICOM'ları yeni ``SeriesInstanceUID`` / ``SOPInstanceUID`` değerleriyle
  yazılır ve kaynak kesitlere ``SourceImageSequence`` ile bağlanır; böylece
  PACS'te orijinal seriyle tanımlayıcı çakışması oluşmaz.

Çalıştırmadan önce
------------------
Model ağırlıklarını (``model_in_5_1.h5``, ``best.pt``) bu betiğin bulunduğu
klasöre koyun; nasıl indirileceği README'de açıklanmıştır:

    pip install -r requirements.txt
    python app/lungscan_gui.py

Ortam değişkenleri
------------------
    LUNGSCAN_WORK_DIR    Çalışma klasörü (varsayılan: bu betiğin klasörü)
    LUNGSCAN_MODEL_DIR   Model ağırlıklarının klasörü (varsayılan: WORK_DIR)
    LUNGSCAN_LOG_LEVEL   DEBUG / INFO / WARNING / ERROR (varsayılan: INFO)

Uyarı
-----
Bu yazılım yalnızca akademik araştırma amaçlıdır; klinik teşhis aracı değildir.
Ayrıntılar için depodaki ``DISCLAIMER.md`` dosyasına bakın.
"""

from __future__ import annotations

import logging
import os
import queue
import shutil
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

import tensorflow as tf
from PIL import Image, ImageTk
from ultralytics import YOLO

import tkinter as tk
from tkinter import Canvas, messagebox
from tkinter.ttk import Progressbar

try:                                        # pydicom >= 3.0
    from pydicom.pixels import apply_voi_lut
except ImportError:                         # pydicom 2.x
    from pydicom.pixel_data_handlers.util import apply_voi_lut


# ==========================================================================
# YAPILANDIRMA
# ==========================================================================
WORK_DIR = Path(os.environ.get("LUNGSCAN_WORK_DIR", Path(__file__).resolve().parent)).expanduser()
MODEL_DIR = Path(os.environ.get("LUNGSCAN_MODEL_DIR", WORK_DIR)).expanduser()

SOURCE_DIR       = WORK_DIR / "kaynak_klasor"     # Girdi: ham DICOM serisi (salt-okunur)
DEST_DIR         = WORK_DIR / "hedef_klas"        # DICOM -> PNG çıktıları
NORMAL_DIR       = WORK_DIR / "normal"            # Sınıflandırıcı: normal kesitler
ABNORMAL_DIR     = WORK_DIR / "abnormal"          # Sınıflandırıcı: anormal kesitler
SEGMENTED_DIR    = WORK_DIR / "segmented"         # YOLO konturlu kesitler
OUTPUT_DIR       = WORK_DIR / "output"            # normal + segmented birleşimi
RESULT_DICOM_DIR = WORK_DIR / "hasta_sonuclari"   # Sonuç DICOM serisi
LOG_PATH         = WORK_DIR / "lungscan.log"

CLASSIFIER_PATH = MODEL_DIR / "model_in_5_1.h5"   # Keras sınıflandırıcı
SEGMENTER_PATH  = MODEL_DIR / "best.pt"           # YOLOv8-seg segmentasyon modeli

# Sınıflandırıcı: eğitimde ImageDataGenerator sınıf adlarını alfabetik sıraladığı
# için indeks 0 = "abnormal", indeks 1 = "normal".
CLASSIFIER_INPUT_SIZE = (256, 256)
ABNORMAL_CLASS_INDEX = 0

# Tek çıkışlı (sigmoid) bir model verilirse kullanılacak karar eşiği.
SIGMOID_ABNORMAL_THRESHOLD = 0.5

# Tek seferde modele verilecek kesit sayısı (tek tek tahminden belirgin hızlıdır).
CLASSIFY_BATCH_SIZE = 32

# Segmentasyon maskesi ikili hâle getirilirken kullanılan eşik.
MASK_THRESHOLD = 0.5

# Kontur kalınlığı (piksel).
CONTOUR_THICKNESS = 1

# Sonuç DICOM'u için SOP sınıfı. True: renkli türev görüntüler için doğru sınıf
# olan Secondary Capture kullanılır. False: orijinal SOP sınıfı korunur (renkli
# piksellerle birlikte teknik olarak geçersizdir, bazı PACS'ler reddeder).
USE_SECONDARY_CAPTURE_SOP_CLASS = True

# Arayüzden değiştirilebilen varsayılan dilim aralığı (0 tabanlı, iki uç dahil).
DEFAULT_START_INDEX = 0
DEFAULT_END_INDEX = 50


# ==========================================================================
# LOGLAMA
# ==========================================================================
logger = logging.getLogger("lungscan")


def _configure_logging() -> None:
    """Konsola ve mümkünse `lungscan.log` dosyasına yazan günlükçüyü kurar."""
    logger.setLevel(os.environ.get("LUNGSCAN_LOG_LEVEL", "INFO").upper())
    logger.propagate = False

    if logger.handlers:                     # yeniden yapılandırmayı önle
        return

    formatter = logging.Formatter("%(asctime)s  %(levelname)-8s %(message)s",
                                  datefmt="%H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    try:                                    # pencersiz (pythonw) çalıştırmada dosya şart
        file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        logger.warning("Günlük dosyası açılamadı: %s", LOG_PATH)


class LungScanError(Exception):
    """Kullanıcıya gösterilebilecek, beklenen hata."""


# ==========================================================================
# DOSYA SİSTEMİ YARDIMCILARI
# ==========================================================================
def clear_directory(folder: Path) -> None:
    """Klasörün **içeriğini** siler, klasörün kendisini korur."""
    folder = Path(folder)
    if not folder.is_dir():
        folder.mkdir(parents=True, exist_ok=True)
        return

    for item in folder.iterdir():
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
        else:
            try:
                item.unlink()
            except OSError:
                logger.debug("Silinemedi: %s", item, exc_info=True)


def has_visible_files(folder: Path) -> bool:
    """Klasörde gizli olmayan en az bir dosya var mı?"""
    folder = Path(folder)
    if not folder.is_dir():
        return False
    return any(p.is_file() and not p.name.startswith(".") for p in folder.iterdir())


def list_pngs(folder: Path) -> list[Path]:
    """Klasördeki PNG dosyalarını ad sırasına göre döndürür (gizli dosyalar hariç)."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(
        (p for p in folder.iterdir()
         if p.is_file() and not p.name.startswith(".") and p.suffix.lower() == ".png"),
        key=lambda p: p.name,
    )


def _to_uint8(array: np.ndarray) -> np.ndarray:
    """Piksel dizisini 8 bite indirger; aykırı değerlere karşı yüzdelik kırpma uygular."""
    if array.dtype == np.uint8:
        return array

    data = np.asarray(array, dtype=np.float32)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros(data.shape, dtype=np.uint8)

    low, high = np.percentile(finite, (0.5, 99.5))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return np.zeros(data.shape, dtype=np.uint8)

    return (np.clip((data - low) / (high - low), 0.0, 1.0) * 255.0).round().astype(np.uint8)


def dicom_pixels_to_image(dataset) -> Image.Image:
    """DICOM piksel verisini görüntülenebilir 8-bit RGB PIL görüntüsüne çevirir."""
    pixels = dataset.pixel_array
    monochrome = str(dataset.get("PhotometricInterpretation", "")).upper().startswith("MONOCHROME")

    if pixels.ndim == 3 and pixels.shape[2] == 3:
        return Image.fromarray(_to_uint8(pixels), mode="RGB")

    if pixels.ndim == 3 and pixels.shape[2] == 1:
        pixels = pixels[:, :, 0]

    if pixels.ndim != 2:
        raise LungScanError(
            f"Desteklenmeyen piksel boyutu: {pixels.shape} "
            f"(PhotometricInterpretation={dataset.get('PhotometricInterpretation')!r})"
        )

    if str(dataset.get("PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        pixels = pixels.max() - pixels

    if monochrome:
        try:
            pixels = apply_voi_lut(pixels, dataset)
        except Exception:
            logger.debug("VOI LUT uygulanamadı; ham piksel değerleri kullanılıyor.",
                         exc_info=True)

    return Image.fromarray(_to_uint8(pixels), mode="L").convert("RGB")


# ==========================================================================
# DICOM OKUMA — kesitleri bul, sırala, PNG'ye çevir
# ==========================================================================
@dataclass(frozen=True)
class Slice:
    """Bir DICOM kesitinin dosya yolu ve sıralama bilgisi."""
    path: Path
    instance_number: int | None

    @property
    def sort_key(self) -> tuple:
        # InstanceNumber yoksa dosya adına göre sıralanır ve sona atılır.
        return (self.instance_number is None,
                0 if self.instance_number is None else self.instance_number,
                self.path.name)


def discover_slices(source_dir: Path) -> list[Slice]:
    """Kaynak klasörü özyinelemeli tarar ve okunabilen DICOM kesitlerini döndürür.

    DICOM olmayan dosyalar (ör. ``.DS_Store``, ``Thumbs.db``) sessizce atlanır.
    Alt klasör yapısı önemli değildir; arşiv nasıl açılmış olursa olsun tüm
    kesitler bulunur.
    """
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise LungScanError(
            f"Kaynak klasör bulunamadı:\n{source_dir}\n\n"
            "Ham DICOM serisini bu klasöre koyup tekrar deneyin."
        )

    slices: list[Slice] = []
    skipped = 0

    for root, _dirs, files in os.walk(source_dir):
        for name in sorted(files):
            if name.startswith("."):
                continue
            path = Path(root) / name
            try:
                header = pydicom.dcmread(str(path), stop_before_pixels=True)
            except Exception:
                skipped += 1
                continue

            if not hasattr(header, "Rows") and not hasattr(header, "PixelData"):
                skipped += 1
                continue

            value = header.get("InstanceNumber")
            try:
                instance_number = int(value) if value is not None else None
            except (TypeError, ValueError):
                instance_number = None

            slices.append(Slice(path=path, instance_number=instance_number))

    if not slices:
        raise LungScanError(
            f"'{source_dir}' klasöründe okunabilir DICOM kesiti bulunamadı.\n\n"
            "Klasörün içeriğini kontrol edin."
        )

    logger.info("%d DICOM kesiti bulundu (%d DICOM olmayan dosya atlandı).",
                len(slices), skipped)
    slices.sort(key=lambda s: s.sort_key)
    return slices


def select_range(slices: Sequence[Slice], start_index: int, end_index: int) -> list[Slice]:
    """[start_index, end_index] aralığını **doğrulayarak** seçer.

    Aralık dışındaki kesitler silinmez; yalnızca sonuç listesine alınmaz.
    """
    if start_index < 0 or end_index < 0:
        raise LungScanError("Başlangıç ve bitiş indeksi negatif olamaz.")

    if start_index > end_index:
        raise LungScanError(
            f"Başlangıç indeksi ({start_index}) bitiş indeksinden ({end_index}) "
            "büyük olamaz.\n\nİndeksler 0'dan başlar ve iki uç da dahildir."
        )

    total = len(slices)
    if start_index >= total:
        raise LungScanError(
            f"Başlangıç indeksi ({start_index}) kesit sayısını aşıyor "
            f"(toplam {total} kesit).\n\nGeçerli aralık: 0 – {total - 1}"
        )

    if end_index >= total:
        logger.warning("Bitiş indeksi (%d) kesit sayısını aşıyor; %d olarak kırpıldı.",
                       end_index, total - 1)
        end_index = total - 1

    return list(slices[start_index:end_index + 1])


def convert_dicom_to_png(start_index: int, end_index: int,
                         progress: Callable[[float, str], None] | None = None) -> int:
    """Seçilen aralıktaki DICOM kesitlerini PNG'ye çevirir.

    ``hedef_klas/`` her çalıştırmada baştan oluşturulur; kaynak klasöre
    dokunulmaz.

    Returns:
        Yazılan PNG sayısı.
    """
    emit = _emitter(progress)
    emit(0.00, "DICOM kesitleri taranıyor...")

    slices = discover_slices(SOURCE_DIR)
    selected = select_range(slices, start_index, end_index)

    logger.info("Aralık seçildi: %d–%d (%d kesit, toplam %d).",
                start_index, end_index, len(selected), len(slices))

    emit(0.05, "Hedef klasör hazırlanıyor...")
    clear_directory(DEST_DIR)

    written = 0
    failed: list[str] = []

    for i, item in enumerate(selected, start=1):
        try:
            dataset = pydicom.dcmread(str(item.path))
            if "PixelData" not in dataset:
                raise LungScanError("PixelData yok")
            image = dicom_pixels_to_image(dataset)
        except Exception as error:
            failed.append(f"{item.path.name}: {error}")
            logger.warning("Kesit okunamadı: %s (%s)", item.path.name, error)
        else:
            image.save(DEST_DIR / f"{item.path.stem}.png", "PNG")
            written += 1

        emit(0.05 + 0.95 * i / len(selected),
             f"DICOM → PNG: {i}/{len(selected)}")

    if failed:
        logger.warning("%d kesit dönüştürülemedi.", len(failed))
    if written == 0:
        raise LungScanError(
            "Hiçbir kesit PNG'ye çevrilemedi.\n\nİlk hatalar:\n" + "\n".join(failed[:5])
        )

    logger.info("%d kesit PNG'ye çevrildi -> %s", written, DEST_DIR)
    return written


# ==========================================================================
# DICOM YAZMA — sonucu orijinal meta verilerle geri dönüştür
# ==========================================================================
#: Renkli türev görüntüye uygulanmaması gereken, monokrom kaynaktan kalan alanlar.
_MONOCHROME_ONLY_KEYWORDS = (
    "WindowCenter", "WindowWidth", "RescaleIntercept", "RescaleSlope",
    "VOILUTFunction", "VOILUTSequence", "ModalityLUTSequence",
)


def build_result_dataset(png_path: Path, source_path: Path,
                         series_uid: str, instance_number: int | None,
                         sop_class_uid: str) -> Dataset:
    """İşlenmiş bir PNG'yi, kaynak DICOM'un meta verileriyle yeni bir DICOM'a yazar."""
    source = pydicom.dcmread(str(source_path))
    result = Dataset()
    result.update(source)                   # meta veri + (üzerine yazılacak) pikseller

    image = Image.open(png_path).convert("RGB")

    result.SOPClassUID = sop_class_uid
    result.SOPInstanceUID = generate_uid()
    result.SeriesInstanceUID = series_uid

    # Pillow'un .size özelliği (genişlik, yükseklik); DICOM'da Rows=yükseklik,
    # Columns=genişlik. (Önceki sürümde bu atama tersti; 512x512'de fark
    # görünmediği için hata gizli kalmıştı.)
    result.Columns, result.Rows = image.size

    result.SamplesPerPixel = 3
    result.PhotometricInterpretation = "RGB"
    result.PlanarConfiguration = 0          # RGB için zorunlu
    result.BitsAllocated = 8
    result.BitsStored = 8
    result.HighBit = 7
    result.PixelRepresentation = 0
    result.PixelData = np.asarray(image).tobytes()

    # Monokrom pencereleme alanları RGB veriye uygulanırsa görüntü bozulur.
    for keyword in _MONOCHROME_ONLY_KEYWORDS:
        if keyword in result:
            del result[keyword]

    result.ImageType = ["DERIVED", "SECONDARY"]
    result.DerivationDescription = (
        "LungScan: kesit siniflandirmasi (normal/anormal) ve lezyon konturlama"
    )
    series_description = str(source.get("SeriesDescription", "") or "").strip()
    result.SeriesDescription = (f"{series_description} [LungScan]"
                                if series_description else "LungScan")

    if instance_number is not None:
        result.InstanceNumber = instance_number

    # Kaynak kesitle bağlantı kur (izlenebilirlik).
    reference = Dataset()
    reference.ReferencedSOPClassUID = source.SOPClassUID
    reference.ReferencedSOPInstanceUID = source.SOPInstanceUID

    # DICOM CID 7204 "Source Image Purposes of Reference" -> 121322 (Source Image)
    purpose = Dataset()
    purpose.CodeValue = "121322"
    purpose.CodingSchemeDesignator = "DCM"
    purpose.CodeMeaning = "Source Image"
    reference.PurposeOfReferenceCodeSequence = [purpose]

    result.SourceImageSequence = [reference]

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = sop_class_uid
    file_meta.MediaStorageSOPInstanceUID = result.SOPInstanceUID
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
    result.file_meta = file_meta

    return result


def convert_pngs_to_dicom(png_folder: Path, source_folder: Path, output_folder: Path,
                          progress: Callable[[float, str], None] | None = None) -> int:
    """Sonuç PNG'lerini, kaynak kesitlerin meta verilerini kullanarak DICOM'a çevirir.

    Dosya eşleştirmesi kök ad üzerinden yapılır: ``<ad>.png`` ↔ ``<ad>.dcm`` ya
    da uzantısız ``<ad>``. Sonuç serisi yeni bir ``SeriesInstanceUID`` alır ve
    orijinal ``InstanceNumber`` sırası korunur.
    """
    emit = _emitter(progress)

    source_by_stem: dict[str, Path] = {}
    for item in discover_slices(source_folder):
        source_by_stem[item.path.stem] = item.path
        if item.path.suffix.lower() == ".dcm":
            source_by_stem[item.path.stem] = item.path

    pngs = list_pngs(png_folder)
    if not pngs:
        raise LungScanError(f"Dönüştürülecek PNG bulunamadı: {output_folder}")

    clear_directory(output_folder)

    sop_class_uid = str(
        SecondaryCaptureImageStorage if USE_SECONDARY_CAPTURE_SOP_CLASS
        else pydicom.dcmread(str(next(iter(source_by_stem.values()))),
                             stop_before_pixels=True).SOPClassUID
    )
    series_uid = generate_uid()

    converted = 0
    missing: list[str] = []

    for i, png_path in enumerate(pngs, start=1):
        source_path = source_by_stem.get(png_path.stem)
        if source_path is None:
            missing.append(png_path.name)
            logger.warning("Kaynak DICOM bulunamadı, atlandı: %s", png_path.name)
            continue

        try:
            header = pydicom.dcmread(str(source_path), stop_before_pixels=True)
        except Exception as error:
            missing.append(f"{png_path.name} ({error})")
            continue

        try:
            value = header.get("InstanceNumber")
            instance_number = int(value) if value is not None else i
        except (TypeError, ValueError):
            instance_number = i

        result = build_result_dataset(png_path, source_path, series_uid,
                                      instance_number, sop_class_uid)
        result.save_as(str(output_folder / f"{png_path.stem}.dcm"), enforce_file_format=True)
        converted += 1

        emit(0.05 + 0.95 * i / len(pngs), f"PNG → DICOM: {i}/{len(pngs)}")

    if missing:
        logger.warning("%d kesit için kaynak DICOM bulunamadı.", len(missing))
    if converted == 0:
        raise LungScanError(
            "Hiçbir kesit DICOM'a çevrilemedi.\n\n"
            "PNG adları ile kaynak DICOM adlarının eşleştiğinden emin olun.\n"
            f"Örnek eşleşmeyen dosya: {missing[0] if missing else '-'}"
        )

    logger.info("%d kesit DICOM'a çevrildi -> %s (SeriesInstanceUID=%s)",
                converted, output_folder, series_uid)
    return converted


# ==========================================================================
# SINIFLANDIRMA
# ==========================================================================
def load_classifier(model_path: Path) -> tuple[tf.keras.Model, str]:
    """Sınıflandırıcıyı yükler ve çıkış biçimini doğrular.

    Modelin çıkış boyutu kontrol edilir: 2 birimli yumuşak maksimum (softmax)
    ya da 1 birimli sigmoid beklenir. Beklenmeyen bir çıkış, kesitlerin
    tamamının aynı sınıfa atanması gibi **sessiz** hatalara yol açabileceği
    için burada açıkça hata verilir.

    Returns:
        (model, mod) — mod ``"softmax"`` veya ``"sigmoid"``.
    """
    model_path = Path(model_path)
    if not model_path.is_file():
        raise LungScanError(
            f"Sınıflandırıcı bulunamadı:\n{model_path}\n\n"
            "Ağırlıkları GitHub Releases'ten indirip bu klasöre koyun "
            "(bkz. README → Model Ağırlıkları)."
        )

    logger.info("Sınıflandırıcı yükleniyor: %s", model_path.name)
    model = tf.keras.models.load_model(str(model_path))

    shape = model.output_shape
    if isinstance(shape, list):
        if len(shape) != 1:
            raise LungScanError(
                f"Sınıflandırıcının birden fazla çıkışı var ({len(shape)}); "
                "tek çıkışlı bir model bekleniyor."
            )
        shape = shape[0]

    if len(shape) != 2:
        raise LungScanError(
            f"Beklenmeyen çıkış biçimi: {shape}. "
            "(batch, sınıf) biçiminde bir sınıflandırıcı bekleniyor."
        )

    units = int(shape[-1])
    if units == 2:
        mode = "softmax"
    elif units == 1:
        mode = "sigmoid"
    else:
        raise LungScanError(
            f"Sınıflandırıcının çıkış birim sayısı {units}; 2 (softmax) veya "
            "1 (sigmoid) bekleniyor."
        )

    logger.info("Sınıflandırıcı hazır (mod: %s, %d parametre).",
                mode, model.count_params())
    return model, mode


def classify_slices(model: tf.keras.Model, mode: str,
                    progress: Callable[[float, str], None] | None = None) -> dict[str, int]:
    """``hedef_klas/`` içindeki kesitleri normal/anormal klasörlerine ayırır.

    Kopyalama yapılır (taşıma değil); böylece adım tekrar çalıştırılabilir.
    Girdi ön işlemesi, modelin eğitiminde kullanılan ``rescale=1./255`` ve
    ``target_size=(256, 256)`` ile birebir aynıdır.

    Returns:
        ``{"normal": n, "abnormal": m}``
    """
    emit = _emitter(progress)

    pngs = list_pngs(DEST_DIR)
    if not pngs:
        raise LungScanError(
            f"'{DEST_DIR}' klasöründe PNG kesit yok.\n\n"
            "Önce 'Start Process' adımını çalıştırın."
        )

    clear_directory(NORMAL_DIR)
    clear_directory(ABNORMAL_DIR)

    counts = {"normal": 0, "abnormal": 0}
    failed: list[str] = []

    for start in range(0, len(pngs), CLASSIFY_BATCH_SIZE):
        batch_paths = pngs[start:start + CLASSIFY_BATCH_SIZE]

        arrays, valid_paths = [], []
        for path in batch_paths:
            try:
                image = Image.open(path).convert("RGB").resize(CLASSIFIER_INPUT_SIZE)
            except Exception as error:
                failed.append(f"{path.name}: {error}")
                logger.warning("Kesit okunamadı: %s (%s)", path.name, error)
                continue
            arrays.append(np.asarray(image, dtype=np.float32) / 255.0)
            valid_paths.append(path)

        if not arrays:
            continue

        predictions = model(np.stack(arrays), training=False).numpy()

        for path, prediction in zip(valid_paths, predictions):
            if mode == "softmax":
                is_abnormal = int(np.argmax(prediction)) == ABNORMAL_CLASS_INDEX
            else:
                is_abnormal = float(prediction[0]) >= SIGMOID_ABNORMAL_THRESHOLD

            target = ABNORMAL_DIR if is_abnormal else NORMAL_DIR
            shutil.copy2(path, target / path.name)
            counts["abnormal" if is_abnormal else "normal"] += 1

        done = min(start + CLASSIFY_BATCH_SIZE, len(pngs))
        emit(0.95 * done / len(pngs), f"Sınıflandırma: {done}/{len(pngs)}")

    if counts["normal"] + counts["abnormal"] == 0:
        raise LungScanError(
            "Hiçbir kesit sınıflandırılamadı.\n\n"
            "İlk hatalar:\n" + "\n".join(failed[:5])
        )

    logger.info("Sınıflandırma tamamlandı: %d normal, %d anormal.",
                counts["normal"], counts["abnormal"])
    return counts


# ==========================================================================
# SEGMENTASYON
# ==========================================================================
def load_segmenter(model_path: Path) -> YOLO:
    """YOLOv8-seg modelini yükler."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise LungScanError(
            f"Segmentasyon modeli bulunamadı:\n{model_path}\n\n"
            "Ağırlıkları GitHub Releases'ten indirip bu klasöre koyun "
            "(bkz. README → Model Ağırlıkları)."
        )

    logger.info("Segmentasyon modeli yükleniyor: %s", model_path.name)
    return YOLO(str(model_path))


def segment_slices(model: YOLO,
                   progress: Callable[[float, str], None] | None = None) -> int:
    """Anormal kesitlerde lezyon konturunu çizer ve ``segmented/`` klasörüne yazar."""
    emit = _emitter(progress)
    clear_directory(SEGMENTED_DIR)

    pngs = list_pngs(ABNORMAL_DIR)
    if not pngs:
        logger.warning("Anormal kesit yok; segmentasyon adımı atlandı.")
        emit(1.0, "Anormal kesit bulunamadı; segmentasyon atlandı.")
        return 0

    drawn_total = 0

    for i, path in enumerate(pngs, start=1):
        image = cv2.imread(str(path))
        if image is None:
            logger.warning("Görüntü okunamadı, atlandı: %s", path.name)
            continue

        height, width = image.shape[:2]
        result = model(image, verbose=False)[0]

        if result.masks is not None:
            for mask in result.masks.data.cpu().numpy():
                resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
                binary = (resized > MASK_THRESHOLD).astype(np.uint8) * 255
                # RETR_EXTERNAL: yalnızca dış sınır çizilir, maske gürültüsünden
                # doğan iç konturlar çizilmez.
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image, contours, -1, (0, 255, 0), CONTOUR_THICKNESS)
                drawn_total += len(contours)

        cv2.imwrite(str(SEGMENTED_DIR / path.name), image)
        emit(i / len(pngs), f"Segmentasyon: {i}/{len(pngs)}")

    logger.info("%d kesit segmentlendi (%d kontur) -> %s",
                len(pngs), drawn_total, SEGMENTED_DIR)
    return len(pngs)


# ==========================================================================
# BİRLEŞTİRME
# ==========================================================================
def merge_results(progress: Callable[[float, str], None] | None = None) -> int:
    """Normal ve işaretli kesitleri tek çıktı klasöründe birleştirir.

    Sonuç serisi **tüm** kesitleri içermelidir; aksi hâlde oluşturulan DICOM
    serisi orijinal kesit sayısıyla eşleşmez ve görüntüleme yazılımında bozuk
    görünür.
    """
    emit = _emitter(progress)
    clear_directory(OUTPUT_DIR)

    copied = 0
    for source in (NORMAL_DIR, SEGMENTED_DIR):
        for path in list_pngs(source):
            shutil.copy2(path, OUTPUT_DIR / path.name)
            copied += 1
        emit(0.5 if source == NORMAL_DIR else 1.0, f"Birleştirme: {copied} kesit")

    expected = len(list_pngs(DEST_DIR))
    if expected and copied != expected:
        logger.warning(
            "Birleşik çıktı (%d kesit), girdiyle (%d kesit) eşleşmiyor. "
            "Bazı kesitler sınıflandırılamamış olabilir.", copied, expected
        )

    logger.info("%d kesit birleştirildi -> %s", copied, OUTPUT_DIR)
    return copied


# ==========================================================================
# İLERLEME YARDIMCISI
# ==========================================================================
def _emitter(progress: Callable[[float, str], None] | None) -> Callable[[float, str], None]:
    """İlerleme geri çağrısını güvenli biçimde saran yardımcı."""
    if progress is None:
        return lambda _fraction, _message: None

    def emit(fraction: float, message: str) -> None:
        try:
            progress(max(0.0, min(1.0, float(fraction))), str(message))
        except Exception:
            logger.debug("İlerleme geri çağrısı başarısız.", exc_info=True)

    return emit


# ==========================================================================
# UÇTAN UCA AKIŞ
# ==========================================================================
#: Uçtan uca akışta adımların ilerleme bütçesindeki payı.
_STEP_BUDGET = (
    ("classify", 0.06, 0.60),
    ("segment", 0.60, 0.90),
    ("merge", 0.90, 0.94),
    ("convert", 0.94, 1.00),
)


def _scaled(progress: Callable[[float, str], None],
            step: str) -> Callable[[float, str], None]:
    """Bir adımın yerel ilerlemesini genel ilerlemeye ölçekler."""
    start, end = next((lo, hi) for name, lo, hi in _STEP_BUDGET if name == step)

    def emit(fraction: float, message: str) -> None:
        progress(start + (end - start) * max(0.0, min(1.0, fraction)), message)

    return emit


def run_prediction(progress: Callable[[float, str], None] | None = None) -> dict[str, int]:
    """Sınıflandırma, segmentasyon, birleştirme ve DICOM dönüşümünü sırayla yürütür.

    Herhangi bir adım başarısız olursa akış **durur** ve hata yukarı taşınır;
    kısmi sonuç DICOM'a çevrilmez.

    Returns:
        Adım bazında üretilen dosya sayıları.
    """
    emit = _emitter(progress)
    summary: dict[str, int] = {}

    emit(0.00, "Sınıflandırıcı hazırlanıyor...")
    classifier, mode = load_classifier(CLASSIFIER_PATH)

    emit(0.06, "Kesitler sınıflandırılıyor...")
    counts = classify_slices(classifier, mode, progress=_scaled(emit, "classify"))
    summary["normal"] = counts["normal"]
    summary["anormal"] = counts["abnormal"]

    emit(0.60, "Segmentasyon modeli hazırlanıyor...")
    segmenter = load_segmenter(SEGMENTER_PATH)

    emit(0.60, "Lezyon konturları çiziliyor...")
    summary["segmentlenen"] = segment_slices(segmenter, progress=_scaled(emit, "segment"))

    emit(0.90, "Kesitler birleştiriliyor...")
    summary["birlesik"] = merge_results(progress=_scaled(emit, "merge"))

    emit(0.94, "Sonuç DICOM'a çevriliyor...")
    summary["dicom"] = convert_pngs_to_dicom(OUTPUT_DIR, SOURCE_DIR, RESULT_DICOM_DIR,
                                             progress=_scaled(emit, "convert"))

    emit(1.00, "Tamamlandı.")
    return summary


# ==========================================================================
# DICOM GÖRÜNTÜLEYİCİ
# ==========================================================================
class DICOMViewer:
    """Basit dilim görüntüleyici: tekerlek/kaydırma, ok tuşları ve pencere boyutu."""

    def __init__(self, master: tk.Toplevel, dicom_folder: Path):
        self.master = master
        self.folder = Path(dicom_folder)
        self.files = self._discover()
        self.index = 0
        self.original_image: Image.Image | None = None
        self.photo_image: ImageTk.PhotoImage | None = None
        self._resize_job: str | None = None

        self.canvas = Canvas(master, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        master.bind("<MouseWheel>", self._on_mousewheel)      # Windows / macOS
        master.bind("<Button-4>", lambda _e: self.show_previous())   # Linux
        master.bind("<Button-5>", lambda _e: self.show_next())       # Linux
        master.bind("<Configure>", self._on_configure)
        master.bind("<Left>", lambda _e: self.show_previous())
        master.bind("<Right>", lambda _e: self.show_next())
        master.bind("<Up>", lambda _e: self.show_previous())
        master.bind("<Down>", lambda _e: self.show_next())
        master.focus_set()

        self.update_image()

    # ------------------------------------------------------------------ kurulum
    def _discover(self) -> list[Path]:
        """Klasördeki okunabilir DICOM dosyalarını InstanceNumber'a göre sıralar.

        DICOM olmayan dosyalar (``.DS_Store`` gibi) ve okunamayan dosyalar
        atlanır; eksik ``InstanceNumber`` dosya adıyla telafi edilir.
        """
        entries: list[tuple[tuple, Path]] = []

        for path in sorted(self.folder.iterdir()):
            if not path.is_file() or path.name.startswith("."):
                continue
            try:
                header = pydicom.dcmread(str(path), stop_before_pixels=True)
            except Exception:
                continue

            value = header.get("InstanceNumber")
            try:
                number = int(value)
            except (TypeError, ValueError):
                number = None

            entries.append(((number is None, number or 0, path.name), path))

        entries.sort(key=lambda item: item[0])
        return [path for _key, path in entries]

    # ------------------------------------------------------------------ çizim
    def _load(self, path: Path) -> Image.Image:
        return dicom_pixels_to_image(pydicom.dcmread(str(path)))

    def update_image(self) -> None:
        if not self.files:
            self.canvas.delete("all")
            self.canvas.create_text(
                max(self.canvas.winfo_width() // 2, 120),
                max(self.canvas.winfo_height() // 2, 80),
                text="Görüntülenecek DICOM dosyası yok.",
                fill="white", font=("Helvetica", 13),
            )
            return

        self.original_image = self._load(self.files[self.index])
        self.display_image()

    def display_image(self) -> None:
        if self.original_image is None:
            return

        canvas_width = max(self.canvas.winfo_width(), 1)
        canvas_height = max(self.canvas.winfo_height(), 1)

        # Pencere henüz yerleşmediyse (canvas 1x1) yeniden boyutlandırma
        # (0, 0) olur ve PIL hata verir; bir sonraki Configure olayını bekleriz.
        if canvas_width <= 1 or canvas_height <= 1:
            return

        original_width, original_height = self.original_image.size
        ratio = min(canvas_width / original_width, canvas_height / original_height)
        new_size = (max(1, int(original_width * ratio)), max(1, int(original_height * ratio)))
        resized = self.original_image.resize(new_size, Image.Resampling.LANCZOS)

        self.canvas.delete("all")
        self.photo_image = ImageTk.PhotoImage(resized)
        self.canvas.create_image(canvas_width / 2, canvas_height / 2,
                                 anchor=tk.CENTER, image=self.photo_image)
        self.canvas.create_text(
            12, 12, anchor=tk.NW, fill="#7dd3fc", font=("Helvetica", 11),
            text=f"{self.files[self.index].name}   [{self.index + 1}/{len(self.files)}]",
        )

    # ------------------------------------------------------------------ olaylar
    def _on_configure(self, _event=None) -> None:
        """Yeniden boyutlandırmayı geciktirir; Configure olay fırtınasını önler."""
        if self._resize_job is not None:
            self.master.after_cancel(self._resize_job)
        self._resize_job = self.master.after(60, self._apply_resize)

    def _apply_resize(self) -> None:
        self._resize_job = None
        if self.original_image is not None:
            self.display_image()

    def _on_mousewheel(self, event) -> None:
        if event.delta > 0:
            self.show_previous()
        elif event.delta < 0:
            self.show_next()

    def show_next(self) -> None:
        if self.index < len(self.files) - 1:
            self.index += 1
            self.update_image()

    def show_previous(self) -> None:
        if self.index > 0:
            self.index -= 1
            self.update_image()

    # ------------------------------------------------------------------ geri uyumluluk
    next_image = show_next
    prev_image = show_previous


# ==========================================================================
# ARAYÜZ
# ==========================================================================
class LungScanApp:
    """Tkinter arayüzü.

    İş parçacığı kuralı: bu sınıfın **tüm** Tk çağrıları ana iş parçacığında
    yapılır. Arka planda çalışan işçi yalnızca ``self._queue`` kuyruğuna mesaj
    bırakır; kuyruk ``_drain_queue`` tarafından ana iş parçacığında boşaltılır.
    """

    POLL_INTERVAL_MS = 100

    def __init__(self, root: tk.Tk):
        self.root = root
        self._queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._busy = False
        self._viewer: tk.Toplevel | None = None

        self._build_ui()
        self.root.after(self.POLL_INTERVAL_MS, self._drain_queue)

    # ------------------------------------------------------------------ arayüz
    def _build_ui(self) -> None:
        root = self.root
        root.title("LungScan — DICOM Processor")
        root.resizable(False, False)

        width, height = 340, 330
        x = int((root.winfo_screenwidth() - width) / 2)
        y = int((root.winfo_screenheight() - height) / 2)
        root.geometry(f"{width}x{height}+{x}+{y}")

        tk.Label(root, text="Start Index (0 tabanlı):").pack(pady=(12, 0))
        self.start_entry = tk.Entry(root, justify="center")
        self.start_entry.insert(0, str(DEFAULT_START_INDEX))
        self.start_entry.pack(fill="x", padx=24)

        tk.Label(root, text="End Index (dahil):").pack(pady=(8, 0))
        self.end_entry = tk.Entry(root, justify="center")
        self.end_entry.insert(0, str(DEFAULT_END_INDEX))
        self.end_entry.pack(fill="x", padx=24)

        self.process_button = tk.Button(root, text="Start Process (DICOM → PNG)",
                                        command=self.on_start_process)
        self.process_button.pack(fill="x", padx=24, pady=(14, 0))

        self.predict_button = tk.Button(root, text="Start Prediction",
                                        command=self.on_start_prediction)
        self.predict_button.pack(fill="x", padx=24, pady=(6, 0))

        self.view_button = tk.Button(root, text="View Result",
                                     command=self.on_view_result)
        self.view_button.pack(fill="x", padx=24, pady=(6, 0))

        self.restart_button = tk.Button(root, text="Restart",
                                        command=self.on_restart)
        self.restart_button.pack(fill="x", padx=24, pady=(6, 0))

        self.progress_var = tk.DoubleVar(master=root, value=0.0)
        self.progress_bar = Progressbar(root, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill="x", padx=24, pady=(14, 4))

        self.status_var = tk.StringVar(master=root, value="Hazır.")
        tk.Label(root, textvariable=self.status_var, anchor="w",
                 fg="#475569", font=("Helvetica", 10)).pack(fill="x", padx=24)

        self._buttons = (self.process_button, self.predict_button,
                         self.view_button, self.restart_button)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        for button in self._buttons:
            button.configure(state=state)

    # ------------------------------------------------------- kuyruk (ana thread)
    def _post(self, kind: str, payload: object = None) -> None:
        """İşçi parçacığından ana parçacığa mesaj bırakır (thread-safe)."""
        self._queue.put((kind, payload))

    def _drain_queue(self) -> None:
        """Kuyruğu boşaltır ve arayüzü günceller.

        Yalnızca ana iş parçacığında çalışır; bu sayede Tkinter'a yapılan tüm
        erişimler tek bir iş parçacığında kalır.
        """
        try:
            while True:
                kind, payload = self._queue.get_nowait()

                if kind == "progress":
                    fraction, message = payload           # type: ignore[misc]
                    self.progress_var.set(float(fraction) * 100.0)
                    self.status_var.set(str(message))

                elif kind == "done":
                    self.progress_var.set(100.0)
                    self.status_var.set("Tamamlandı.")
                    self._set_busy(False)
                    messagebox.showinfo("LungScan", str(payload))

                elif kind == "error":
                    self.progress_var.set(0.0)
                    self.status_var.set("Hata.")
                    self._set_busy(False)
                    messagebox.showerror("LungScan — Hata", str(payload))

                elif kind == "info":
                    messagebox.showinfo("LungScan", str(payload))

        except queue.Empty:
            pass
        finally:
            self.root.after(self.POLL_INTERVAL_MS, self._drain_queue)

    # ------------------------------------------------------------------ işçiler
    def _run_in_background(self, job: Callable[[], str]) -> None:
        """`job`'u ayrı iş parçacığında çalıştırır; sonucu kuyruğa bırakır."""
        if self._busy:
            return
        self._set_busy(True)
        self.progress_var.set(0.0)

        def worker() -> None:
            try:
                self._post("done", job())
            except LungScanError as error:
                logger.warning("Akış durduruldu: %s", error)
                self._post("error", str(error))
            except Exception:
                logger.exception("Beklenmeyen hata.")
                self._post("error",
                           "Beklenmeyen bir hata oluştu. Ayrıntılar lungscan.log "
                           "dosyasında.\n\n" + traceback.format_exc(limit=3))

        threading.Thread(target=worker, daemon=True).start()

    def _read_indices(self) -> tuple[int, int]:
        """Arayüzdeki indeks alanlarını okur ve doğrular."""
        raw_start = self.start_entry.get().strip()
        raw_end = self.end_entry.get().strip()
        try:
            start_index = int(raw_start)
            end_index = int(raw_end)
        except ValueError:
            raise LungScanError(
                "İndeks alanları tam sayı olmalıdır.\n\n"
                f"Girilen değerler: Başlangıç = {raw_start!r}, Bitiş = {raw_end!r}"
            ) from None

        if start_index > end_index:
            raise LungScanError(
                f"Başlangıç indeksi ({start_index}) bitiş indeksinden ({end_index}) "
                "büyük olamaz.\n\nİndeksler 0'dan başlar ve iki uç da dahildir."
            )
        if start_index < 0:
            raise LungScanError("Başlangıç indeksi negatif olamaz.")

        return start_index, end_index

    # ------------------------------------------------------------------ eylemler
    def on_start_process(self) -> None:
        """1. adım: DICOM → PNG. Kaynak klasöre dokunulmaz."""
        try:
            start_index, end_index = self._read_indices()
        except LungScanError as error:
            messagebox.showerror("LungScan — Hata", str(error))
            return

        def job() -> str:
            progress = lambda fraction, message: self._post("progress", (fraction, message))
            written = convert_dicom_to_png(start_index, end_index, progress)
            return (f"{written} kesit PNG'ye çevrildi.\n\n"
                    f"Kaynak: {SOURCE_DIR}\nHedef: {DEST_DIR}\n\n"
                    "Kaynak klasördeki DICOM dosyalarına dokunulmadı. "
                    "Şimdi 'Start Prediction' adımını çalıştırabilirsiniz.")

        self._run_in_background(job)

    def on_start_prediction(self) -> None:
        """2-5. adımlar: sınıflandırma, segmentasyon, birleştirme, DICOM."""

        def job() -> str:
            progress = lambda fraction, message: self._post("progress", (fraction, message))
            summary = run_prediction(progress)
            return (
                "İşlem tamamlandı.\n\n"
                f"Normal kesit        : {summary.get('normal', 0)}\n"
                f"Anormal kesit       : {summary.get('anormal', 0)}\n"
                f"Lezyon içeren kesit : {summary.get('segmentlenen', 0)}\n"
                f"Birleşik çıktı      : {summary.get('birlesik', 0)}\n"
                f"Sonuç DICOM         : {summary.get('dicom', 0)}\n\n"
                f"Çıktı klasörü: {RESULT_DICOM_DIR}\n\n"
                "Sonuçları 'View Result' ile inceleyebilirsiniz."
            )

        self._run_in_background(job)

    def on_restart(self) -> None:
        """Ara klasörleri temizler. **Kaynak klasöre dokunmaz.**"""
        if self._busy:
            return

        if not messagebox.askyesno(
            "LungScan — Restart",
            "Ara klasörler temizlenecek:\n"
            f"• {DEST_DIR.name}/\n• {NORMAL_DIR.name}/\n• {ABNORMAL_DIR.name}/\n"
            f"• {SEGMENTED_DIR.name}/\n• {OUTPUT_DIR.name}/\n• {RESULT_DICOM_DIR.name}/\n\n"
            f"Kaynak klasör ({SOURCE_DIR.name}/) korunur.\n\nDevam edilsin mi?",
        ):
            return

        try:
            for folder in (DEST_DIR, NORMAL_DIR, ABNORMAL_DIR,
                           SEGMENTED_DIR, OUTPUT_DIR, RESULT_DICOM_DIR):
                clear_directory(folder)
        except OSError as error:
            messagebox.showerror("LungScan — Hata", f"Klasörler temizlenemedi:\n{error}")
            return

        self.progress_var.set(0.0)
        self.status_var.set("Ara klasörler temizlendi.")
        logger.info("Restart: ara klasörler temizlendi.")

    def on_view_result(self) -> None:
        """Sonuç DICOM serisini görüntüleyicide açar."""
        if self._viewer is not None and self._viewer.winfo_exists():
            self._viewer.lift()
            self._viewer.focus_set()
            return

        if not has_visible_files(RESULT_DICOM_DIR):
            messagebox.showinfo(
                "LungScan — Viewer",
                f"Gösterilecek sonuç bulunamadı:\n{RESULT_DICOM_DIR}\n\n"
                "Önce 'Start Prediction' adımını çalıştırın.",
            )
            return

        self._viewer = tk.Toplevel(self.root)
        self._viewer.title("LungScan — DICOM Viewer")
        self._viewer.geometry("620x620")
        DICOMViewer(self._viewer, RESULT_DICOM_DIR)


# ==========================================================================
# GİRİŞ NOKTASI
# ==========================================================================
def main() -> int:
    _configure_logging()
    logger.info("LungScan arayüzü başlatılıyor (çalışma klasörü: %s).", WORK_DIR)
    logger.info("Model klasörü: %s", MODEL_DIR)

    root = tk.Tk()
    LungScanApp(root)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        logger.info("Kullanıcı tarafından kapatıldı.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
