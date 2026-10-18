#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DICOM katmanı testleri.

Arayüzün veri güvenliği açısından kritik davranışlarını sınar:

* Kaynak klasördeki DICOM dosyalarına **hiçbir koşulda** yazılmaz/silinmez.
* Geçersiz indeks aralıkları hata verir; sessizce veri kaybına yol açmaz.
* DICOM olmayan dosyalar (``.DS_Store``, ``Thumbs.db``) taranırken atlanır.
* Sonuç DICOM'ları doğru ``Rows``/``Columns``, yeni ``SeriesInstanceUID`` /
  ``SOPInstanceUID`` ve RGB'ye uygun piksel alanlarıyla yazılır.
* Kısmi başarısızlık sessizce yutulmaz.

Ağır bağımlılıklar (TensorFlow, OpenCV, Ultralytics) taklit edilir; bu sayede
testler yalnızca ``numpy``, ``pillow`` ve ``pydicom`` ile çalışır::

    python tests/test_dicom_katmani.py
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
import traceback
import types
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image
from pydicom.dataset import FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = REPO_ROOT / "app" / "lungscan_gui.py"

#: Sentetik test kesitlerinde kullanılan SOP sınıfı (CT Image Storage).
TEST_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.2"

_results: list[tuple[bool, str]] = []


# ==========================================================================
# Yardımcılar
# ==========================================================================
def _stub_heavy_dependencies() -> None:
    """TensorFlow / OpenCV / Ultralytics yerine boş modüller koyar."""
    for name in ("cv2", "tensorflow", "ultralytics"):
        if name in sys.modules:
            continue

        stub = types.ModuleType(name)
        if name == "cv2":
            stub.INTER_LINEAR = 1
            stub.RETR_EXTERNAL = 0
            stub.CHAIN_APPROX_SIMPLE = 1
            for function in ("imread", "imwrite", "resize", "findContours", "drawContours"):
                setattr(stub, function, lambda *args, **kwargs: None)
        elif name == "tensorflow":
            stub.keras = types.SimpleNamespace(
                models=types.SimpleNamespace(load_model=lambda *a, **k: None),
                Model=object,
            )
        else:
            stub.YOLO = object
        sys.modules[name] = stub


def _load_gui_module():
    """`app/lungscan_gui.py` dosyasını ağır bağımlılıklar taklit edilerek yükler."""
    _stub_heavy_dependencies()
    spec = importlib.util.spec_from_file_location("lungscan_gui", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["lungscan_gui"] = module
    spec.loader.exec_module(module)
    return module


def write_slice(path: Path, instance_number: int, size: tuple[int, int] = (64, 48),
                rgb: bool = True) -> None:
    """Gerçekçi bir kaynak kesit yazar (RGB ya da pencerelemeli monokrom)."""
    width, height = size
    rng = np.random.default_rng(instance_number)
    if rgb:
        pixels = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    else:
        pixels = rng.integers(0, 4096, (height, width), dtype=np.uint16)

    dataset = pydicom.Dataset()
    dataset.SOPClassUID = TEST_SOP_CLASS
    dataset.SOPInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    dataset.StudyInstanceUID = generate_uid()
    dataset.PatientName = "TEST^HASTA"
    dataset.PatientID = "TEST-001"
    dataset.Modality = "CT"
    dataset.InstanceNumber = instance_number
    dataset.SeriesDescription = "PET/CT fused"
    dataset.Rows, dataset.Columns = height, width
    dataset.SamplesPerPixel = 3 if rgb else 1
    dataset.PhotometricInterpretation = "RGB" if rgb else "MONOCHROME2"
    dataset.PlanarConfiguration = 0
    dataset.BitsAllocated = 8 if rgb else 16
    dataset.BitsStored = 8 if rgb else 12
    dataset.HighBit = 7 if rgb else 11
    dataset.PixelRepresentation = 0
    if not rgb:
        dataset.WindowCenter = 2048
        dataset.WindowWidth = 4096
        dataset.RescaleIntercept = -1024
        dataset.RescaleSlope = 1
    dataset.PixelData = pixels.tobytes()

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = TEST_SOP_CLASS
    meta.MediaStorageSOPInstanceUID = dataset.SOPInstanceUID
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset.file_meta = meta
    dataset.save_as(str(path), enforce_file_format=True)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    _results.append((True, message))


def heading(text: str) -> None:
    print(f"\n{text}\n{'-' * len(text)}")


# ==========================================================================
# Testler
# ==========================================================================
def test_discovery(gui, source_dir: Path) -> list:
    """DICOM olmayan dosyalar atlanmalı, tüm kesitler bulunmalı."""
    heading("1) Kaynak klasör taranması")

    (source_dir / "DICOM" / "alt").mkdir(parents=True, exist_ok=True)
    (source_dir / ".DS_Store").write_bytes(b"cop")
    (source_dir / "Thumbs.db").write_bytes(b"cop")

    for number in range(1, 9):
        target = source_dir / ("DICOM/alt" if number > 4 else "")
        write_slice(target / f"s{number:03d}", number)
    write_slice(source_dir / "mono001", 99, rgb=False)

    slices = gui.discover_slices(source_dir)
    print(f"  bulunan kesit        : {len(slices)} (beklenen 9)")
    print(f"  InstanceNumber sırası: {[s.instance_number for s in slices]}")

    check(len(slices) == 9, "DICOM olmayan dosyalar atlanmadı veya kesit kaybı var")
    check([s.instance_number for s in slices] == list(range(1, 9)) + [99],
          "kesitler InstanceNumber'a göre sıralanmadı")
    return slices


def test_invalid_ranges(gui, source_dir: Path) -> None:
    """Geçersiz aralıklar hata vermeli ve kaynak dosyalara dokunmamalı."""
    heading("2) Geçersiz aralıklar (veri kaybı olmamalı)")

    before = sorted(p.name for p in source_dir.rglob("*") if p.is_file())

    cases = ((100, 50, "ters aralık"), (-1, 5, "negatif indeks"), (500, 600, "aralık dışı"))
    for start, end, label in cases:
        try:
            gui.convert_dicom_to_png(start, end)
        except gui.LungScanError as error:
            print(f"  {label:<16} -> LungScanError: {str(error).splitlines()[0]}")
            _results.append((True, f"{label} reddedildi"))
        else:
            raise AssertionError(f"{label} için hata bekleniyordu!")

    after = sorted(p.name for p in source_dir.rglob("*") if p.is_file())
    print(f"  kaynak dosya sayısı  : {len(before)} -> {len(after)}")
    check(before == after, "geçersiz aralık kaynak dosyaları sildi veya değiştirdi")


def test_valid_conversion(gui, source_dir: Path) -> None:
    """Geçerli aralık PNG üretmeli, kaynağı korumalı."""
    heading("3) Geçerli aralık (DICOM -> PNG)")

    before = sorted(p.name for p in source_dir.rglob("*") if p.is_file())
    written = gui.convert_dicom_to_png(2, 6)
    after = sorted(p.name for p in source_dir.rglob("*") if p.is_file())

    print(f"  yazılan PNG          : {written} (beklenen 5)")
    print(f"  kaynak korundu       : {before == after}")

    check(written == 5, "beklenen sayıda PNG üretilmedi")
    check(before == after, "geçerli aralıkta kaynak dosyalar değişti")
    check((gui.DEST_DIR / "s003.png").is_file(), "s003.png üretilmedi")


def test_monochrome_input(gui, source_dir: Path) -> None:
    """16-bit pencerelemeli monokrom kesit 8-bit RGB görüntüye çevrilmeli."""
    heading("4) Monokrom (16-bit) kesit dönüşümü")

    dataset = pydicom.dcmread(str(source_dir / "mono001"))
    image = gui.dicom_pixels_to_image(dataset)
    array = np.asarray(image)

    print(f"  PIL biçimi           : {image.mode} {image.size}")
    print(f"  piksel aralığı       : {array.min()}–{array.max()}")

    check(image.mode == "RGB", "monokrom kesit RGB'ye çevrilmedi")
    check(array.dtype == np.uint8, "piksel verisi 8 bit değil")


def test_png_to_dicom(gui, source_dir: Path) -> None:
    """Sonuç DICOM'ları doğru alanlar ve yeni kimliklerle yazılmalı."""
    heading("5) PNG -> DICOM dönüşümü")

    gui.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(gui.DEST_DIR / "s003.png", gui.OUTPUT_DIR / "s003.png")
    # Kare olmayan bir görüntü: Rows/Columns karışıklığını yakalar.
    Image.new("RGB", (64, 32), (10, 20, 30)).save(gui.OUTPUT_DIR / "s004.png")
    (gui.OUTPUT_DIR / "notlar.txt").write_text("x")     # PNG olmayan dosya sayılmamalı

    converted = gui.convert_pngs_to_dicom(gui.OUTPUT_DIR, source_dir, gui.RESULT_DICOM_DIR)
    print(f"  dönüştürülen kesit   : {converted} (beklenen 2)")
    check(converted == 2, "beklenen sayıda kesit dönüştürülmedi")

    source = pydicom.dcmread(str(source_dir / "s003"))
    result = pydicom.dcmread(str(gui.RESULT_DICOM_DIR / "s003.dcm"))
    rectangle = pydicom.dcmread(str(gui.RESULT_DICOM_DIR / "s004.dcm"))

    expected_fields = (
        ("PhotometricInterpretation", result.PhotometricInterpretation, "RGB"),
        ("PlanarConfiguration", result.PlanarConfiguration, 0),
        ("SamplesPerPixel", result.SamplesPerPixel, 3),
        ("BitsAllocated", result.BitsAllocated, 8),
        ("PixelRepresentation", result.PixelRepresentation, 0),
        ("InstanceNumber", result.InstanceNumber, source.InstanceNumber),
    )
    for label, actual, expected in expected_fields:
        print(f"  {label:<26}: {actual}")
        check(actual == expected, f"{label} beklenen değerde değil")

    print(f"  kare olmayan görüntü         : Columns={rectangle.Columns}, Rows={rectangle.Rows}")
    check((rectangle.Columns, rectangle.Rows) == (64, 32),
          "Rows/Columns ters yazılıyor (Pillow .size sırasına dikkat)")

    check(source.SeriesInstanceUID != result.SeriesInstanceUID,
          "SeriesInstanceUID yeniden üretilmedi (PACS'te çakışma riski)")
    check(source.SOPInstanceUID != result.SOPInstanceUID,
          "SOPInstanceUID yeniden üretilmedi (PACS'te çakışma riski)")
    check(result.SeriesInstanceUID == rectangle.SeriesInstanceUID,
          "aynı çalıştırmadaki kesitler farklı serilere dağıldı")

    reference = result.SourceImageSequence[0]
    check(reference.ReferencedSOPInstanceUID == source.SOPInstanceUID,
          "kaynak kesit referansı eksik (izlenebilirlik)")
    check(list(result.ImageType) == ["DERIVED", "SECONDARY"],
          "ImageType türev görüntü olarak işaretlenmedi")

    stale = [k for k in ("WindowCenter", "WindowWidth", "RescaleIntercept", "RescaleSlope")
             if k in result]
    check(not stale, f"RGB veride monokrom pencereleme alanları kaldı: {stale}")

    pixels = result.pixel_array
    print(f"  okunan piksel biçimi : {pixels.shape} {pixels.dtype}")
    check(pixels.shape == (source.Rows, source.Columns, 3), "piksel boyutu kaynakla uyuşmuyor")


def test_missing_models(gui) -> None:
    """Eksik model dosyaları anlaşılır hata vermeli."""
    heading("6) Eksik model dosyaları")

    for loader, label in ((gui.load_classifier, "sınıflandırıcı"),
                          (gui.load_segmenter, "segmentasyon")):
        try:
            loader(Path("/olmayan/klasor/model.dosyasi"))
        except gui.LungScanError as error:
            print(f"  {label:<14} -> {str(error).splitlines()[0]}")
            check("bulunamadı" in str(error), f"{label} için anlaşılır hata yok")
        else:
            raise AssertionError(f"{label} için hata bekleniyordu")


def test_empty_inputs(gui, tmp: Path) -> None:
    """Boş klasörler sessizce başarısız olmamalı."""
    heading("7) Boş girdi klasörleri")

    empty = tmp / "bos_klasor"
    empty.mkdir(exist_ok=True)
    try:
        gui.discover_slices(empty)
    except gui.LungScanError as error:
        print(f"  boş kaynak klasör -> {str(error).splitlines()[0]}")
        _results.append((True, "boş kaynak klasör reddedildi"))
    else:
        raise AssertionError("boş klasör için hata bekleniyordu")


def test_to_uint8(gui) -> None:
    """Piksel normalizasyonu taşma ve sıfıra bölme hatası üretmemeli."""
    heading("8) Piksel normalizasyonu")

    already_8bit = np.array([[0, 128, 255]], dtype=np.uint8)
    check(np.array_equal(gui._to_uint8(already_8bit), already_8bit),
          "8 bit girdi değiştirilmemeli")

    wide = gui._to_uint8(np.array([[0, 1000, 4000]], dtype=np.uint16))
    print(f"  16 bit [0,1000,4000] -> {wide.tolist()}")
    check(wide.dtype == np.uint8 and wide.max() == 255, "16 bit veri 8 bite indirgenmedi")

    flat = gui._to_uint8(np.full((4, 4), 7, dtype=np.uint16))
    check(flat.max() == 0, "sabit dizide sıfıra bölme hatası")


# ==========================================================================
# Çalıştırıcı
# ==========================================================================
def main() -> int:
    if not MODULE_PATH.is_file():
        print(f"HATA: {MODULE_PATH} bulunamadı.", file=sys.stderr)
        return 2

    gui = _load_gui_module()

    tmp = Path(tempfile.mkdtemp(prefix="lungscan_test_"))
    try:
        # Çalışma klasörlerini geçici dizine yönlendir.
        for attribute, name in (("SOURCE_DIR", "kaynak_klasor"), ("DEST_DIR", "hedef_klas"),
                                ("NORMAL_DIR", "normal"), ("ABNORMAL_DIR", "abnormal"),
                                ("SEGMENTED_DIR", "segmented"), ("OUTPUT_DIR", "output"),
                                ("RESULT_DICOM_DIR", "hasta_sonuclari")):
            setattr(gui, attribute, tmp / name)
        gui.WORK_DIR = tmp
        gui.SOURCE_DIR.mkdir(parents=True, exist_ok=True)

        test_discovery(gui, gui.SOURCE_DIR)
        test_invalid_ranges(gui, gui.SOURCE_DIR)
        test_valid_conversion(gui, gui.SOURCE_DIR)
        test_monochrome_input(gui, gui.SOURCE_DIR)
        test_png_to_dicom(gui, gui.SOURCE_DIR)
        test_missing_models(gui)
        test_empty_inputs(gui, tmp)
        test_to_uint8(gui)

    except AssertionError as error:
        print(f"\nBAŞARISIZ: {error}")
        traceback.print_exc()
        return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'=' * 62}")
    print(f"TÜM TESTLER GEÇTİ — {len(_results)} kontrol")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
