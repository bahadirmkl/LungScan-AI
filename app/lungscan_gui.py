#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LungScan - Akciger BT (DICOM) goruntulerinde kanser tespiti ve segmentasyonu.

Bu arayuz su akisi yurutur:
    1) DICOM serisini temizle ve secilen aralikta PNG'ye cevir
    2) Keras siniflandirici ile kesitleri normal / anormal olarak ayir
    3) YOLOv8-seg modeli ile anormal kesitlerde lezyonu segmente et
    4) Normal + segmente kesitleri birlestir
    5) Sonucu tekrar DICOM'a cevir
    6) Dahili DICOM goruntuleyici ile sonucu incele

Calistirmadan once:
    pip install -r requirements.txt

Ortam degiskenleri (istege bagli):
    LUNGSCAN_WORK_DIR   Calisma klasoru (varsayilan: bu betigin bulundugu klasor)
    LUNGSCAN_MODEL_DIR  Model agirliklarinin klasoru (varsayilan: LUNGSCAN_WORK_DIR)

Model agirliklari GitHub Releases uzerinden indirilir; README'ye bakin.
"""

import copy
import os
import shutil
import traceback
import threading
from pathlib import Path

import cv2
import numpy as np
import pydicom
import tensorflow as tf
from PIL import Image, ImageTk
from pydicom.pixel_data_handlers.util import apply_voi_lut
from ultralytics import YOLO

import tkinter as tk
from tkinter import Canvas, filedialog, messagebox
from tkinter.ttk import Progressbar


# --------------------------------------------------------------------------
# YAPILANDIRMA
# --------------------------------------------------------------------------
WORK_DIR = Path(os.environ.get("LUNGSCAN_WORK_DIR", Path(__file__).resolve().parent)).expanduser()
MODEL_DIR = Path(os.environ.get("LUNGSCAN_MODEL_DIR", WORK_DIR)).expanduser()

SOURCE_DIR      = WORK_DIR / "kaynak_klasor"      # Ham DICOM serisi buraya konur
DEST_DIR        = WORK_DIR / "hedef_klas"         # DICOM -> PNG ciktilari
NORMAL_DIR      = WORK_DIR / "normal"             # Siniflandirici: normal kesitler
ABNORMAL_DIR    = WORK_DIR / "abnormal"           # Siniflandirici: anormal kesitler
SEGMENTED_DIR   = WORK_DIR / "segmented"          # YOLO konturlu kesitler
OUTPUT_DIR      = WORK_DIR / "output"             # normal + segmented birlesimi
RESULT_DICOM_DIR = WORK_DIR / "hasta_sonuclari"   # Sonuc DICOM serisi

# Keras siniflandirici (normal / anormal)
CLASSIFIER_PATH = MODEL_DIR / "model_in_5_1.h5"
# YOLOv8-seg lezyon segmentasyon modeli
SEGMENTER_PATH  = MODEL_DIR / "best.pt"

# Siniflandirici giris boyutu ve esik kurali
CLASSIFIER_INPUT_SIZE = (256, 256)
ABNORMAL_CLASS_INDEX = 0     # argmax == 0 -> anormal

# Segmentasyon maskesi esigi
MASK_THRESHOLD = 0.5

# Islenecek DICOM dilim araligi (arayuzden de girilebilir)
DEFAULT_START_INDEX = 0
DEFAULT_END_INDEX = 50


# --------------------------------------------------------------------------
# DICOM yardimcilari
# --------------------------------------------------------------------------
def clean_and_move_dicom_files(source_folder):
    """Kaynak klasordeki DICOM disi her seyi siler, DICOM/ altini kok'e tasir."""
    source_folder = str(source_folder)
    items = os.listdir(source_folder)
    for item in items:
        item_path = os.path.join(source_folder, item)
        if item != "DICOM" and os.path.isdir(item_path):
            shutil.rmtree(item_path)
        elif item != "DICOM" and os.path.isfile(item_path):
            os.remove(item_path)

    for root, dirs, files in os.walk(os.path.join(source_folder, "DICOM"), topdown=False):
        if not dirs:
            for file in files:
                shutil.move(os.path.join(root, file), source_folder)
            break

    shutil.rmtree(os.path.join(source_folder, "DICOM"))


def dicom_to_png_filtered(source_folder, destination_folder, start_index, end_index):
    """DICOM serisini InstanceNumber'a gore siralar, [start,end] araligini PNG yazar."""
    clean_and_move_dicom_files(source_folder)

    files = [os.path.join(source_folder, f) for f in os.listdir(source_folder)
             if os.path.isfile(os.path.join(source_folder, f))]

    def sort_key(filepath):
        dicom_image = pydicom.dcmread(filepath)
        return int(dicom_image.get('InstanceNumber', 9999))

    files_sorted = sorted(files, key=sort_key)

    os.makedirs(destination_folder, exist_ok=True)

    for i, file_path in enumerate(files_sorted, start=0):
        if start_index <= i <= end_index:
            dicom_image = pydicom.dcmread(file_path)
            if 'PixelData' in dicom_image:
                image = Image.fromarray(dicom_image.pixel_array)
                original_file_name = os.path.splitext(os.path.basename(file_path))[0] + ".png"
                image.save(os.path.join(destination_folder, original_file_name), "PNG")
            else:
                print(f"No PixelData found in: {file_path}")
        else:
            os.remove(file_path)


def is_color_dicom(dicom_path):
    ds = pydicom.dcmread(dicom_path)
    return ds.PhotometricInterpretation.startswith('RGB')


def convert_to_color(png_image):
    return png_image.convert('RGB')


def resize_image(png_image, width, height):
    return png_image.resize((width, height))


def create_dicom_from_png(png_image, original_dicom, instance_number):
    """Orijinal DICOM meta verisini koruyarak PNG piksellerini yeni DICOM'a yazar."""
    new_dicom = copy.deepcopy(original_dicom)

    new_dicom.PixelData = np.array(png_image).tobytes()

    new_dicom.Rows, new_dicom.Columns = png_image.size

    if not is_color_dicom(original_dicom.filename):
        new_dicom.PhotometricInterpretation = 'RGB'

    new_dicom.SamplesPerPixel = 3
    new_dicom.BitsAllocated = 8
    new_dicom.BitsStored = 8
    new_dicom.HighBit = 7
    new_dicom.PixelRepresentation = 0

    new_dicom.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    new_dicom.InstanceNumber = instance_number

    return new_dicom


def convert_png_to_dicom(png_folder, original_dicom_folder, output_folder):
    """PNG sonuclarini InstanceNumber sirasini koruyarak tekrar DICOM'a cevirir."""
    instance_numbers = {}

    for dicom_filename in os.listdir(original_dicom_folder):
        dicom_path = os.path.join(original_dicom_folder, dicom_filename)
        original_dicom = pydicom.dcmread(dicom_path)
        instance_number = original_dicom.get('InstanceNumber', 0)
        instance_numbers[dicom_filename] = instance_number

    sorted_png_filenames = sorted(
        os.listdir(png_folder),
        key=lambda fname: instance_numbers.get(fname.replace('.png', ''), 0)
    )

    for i, png_filename in enumerate(sorted_png_filenames, start=1):
        if png_filename.lower().endswith('.png'):
            png_path = os.path.join(png_folder, png_filename)
            dicom_filename_without_ext = png_filename.split('.')[0]
            original_dicom_path = os.path.join(original_dicom_folder, dicom_filename_without_ext)
            output_dicom_path = os.path.join(output_folder, dicom_filename_without_ext)

            original_dicom = pydicom.dcmread(original_dicom_path)

            png_image = Image.open(png_path).convert('RGB')
            png_image = resize_image(png_image, 512, 512)

            new_dicom = create_dicom_from_png(png_image, original_dicom, i)
            new_dicom.save_as(output_dicom_path)

            print(f"Converted {png_filename} to DICOM and saved as {output_dicom_path} "
                  f"with InstanceNumber {i}")


# --------------------------------------------------------------------------
# Siniflandirma + segmentasyon
# --------------------------------------------------------------------------
def update_progress_bar(value):
    """Ana thread'de ilerleme cubugunu gunceller."""
    progress_var.set(value)
    root.update_idletasks()


def predict_and_move_images(model_path, destination_folder,
                            target_normal_folder, target_abnormal_folder, progress_var):
    """Keras siniflandirici ile kesitleri normal/abnormal klasorlerine ayirir."""
    try:
        print(f"Loading model from: {model_path}")
        model = tf.keras.models.load_model(model_path)

        image_files = [f for f in os.listdir(destination_folder) if f.endswith('.png')]
        steps_per_image = 35 / max(len(image_files), 1)

        for index, image_file in enumerate(image_files):
            image_path = os.path.join(destination_folder, image_file)
            img = Image.open(image_path).resize(CLASSIFIER_INPUT_SIZE)
            img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
            prediction = model.predict(img_array)

            if np.argmax(prediction) == ABNORMAL_CLASS_INDEX:
                target_folder = target_abnormal_folder
            else:
                target_folder = target_normal_folder

            os.makedirs(target_folder, exist_ok=True)
            os.replace(image_path, os.path.join(target_folder, image_file))

            root.after(0, update_progress_bar, 5 + (index + 1) * steps_per_image)

    except Exception as e:
        messagebox.showerror("Error", str(e))
        print(str(e))


def run_segmentation(progress_var):
    """Anormal kesitlerde YOLOv8-seg ile lezyon konturunu cizer."""
    yolo_model = YOLO(str(SEGMENTER_PATH))
    os.makedirs(SEGMENTED_DIR, exist_ok=True)

    image_files = os.listdir(ABNORMAL_DIR)
    per_image_increment = 50 / max(len(image_files), 1)

    for index, image_file in enumerate(image_files):
        if image_file.endswith('.png'):
            img_path = os.path.join(ABNORMAL_DIR, image_file)
            img = cv2.imread(img_path)
            H, W, _ = img.shape

            results = yolo_model(img)
            progress_var.set(40 + (index + 1) * per_image_increment)
            root.update_idletasks()

            if results[0].masks is not None:
                predicted_masks = results[0].masks.data.cpu().numpy()
                for mask in predicted_masks:
                    mask = cv2.resize(mask, (W, H))
                    _, mask_thresh = cv2.threshold(mask, MASK_THRESHOLD, 255, cv2.THRESH_BINARY)
                    mask_thresh = mask_thresh.astype(np.uint8)
                    contours, _ = cv2.findContours(mask_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(img, contours, -1, (0, 255, 0), 1)

            save_path = os.path.join(SEGMENTED_DIR, image_file)
            cv2.imwrite(save_path, img)


def merge_outputs(progress_var):
    """Normal ve segmente kesitleri tek cikti klasorunde birlestirir."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for src_dir in (NORMAL_DIR, SEGMENTED_DIR):
        for filename in os.listdir(src_dir):
            if filename.endswith('.png'):
                shutil.copy(os.path.join(src_dir, filename), os.path.join(OUTPUT_DIR, filename))

    print("All images have been copied to the 'output' folder.")
    progress_var.set(95)


# --------------------------------------------------------------------------
# Akis kontrolu
# --------------------------------------------------------------------------
def start_process():
    """1. adim: DICOM -> PNG donusumu."""
    try:
        start_index = int(start_index_entry.get())
        end_index = int(end_index_entry.get())

        dicom_to_png_filtered(str(SOURCE_DIR), str(DEST_DIR), start_index, end_index)

        messagebox.showinfo("Success", "Process completed successfully.")
    except Exception as e:
        messagebox.showerror("Error", str(e))


def start_prediction():
    """2-5. adimlar: siniflandirma, segmentasyon, birlestirme, DICOM'a geri donusum."""
    progress_var.set(0)
    root.update_idletasks()

    try:
        model = tf.keras.models.load_model(str(CLASSIFIER_PATH))
        progress_var.set(5)
        root.update_idletasks()

        predict_and_move_images(str(CLASSIFIER_PATH), str(DEST_DIR),
                                str(NORMAL_DIR), str(ABNORMAL_DIR), progress_var)

        run_segmentation(progress_var)
        merge_outputs(progress_var)

    except Exception as e:
        messagebox.showerror("Error", traceback.format_exc())

    try:
        os.makedirs(RESULT_DICOM_DIR, exist_ok=True)
        convert_png_to_dicom(str(OUTPUT_DIR), str(SOURCE_DIR), str(RESULT_DICOM_DIR))

        progress_var.set(100)
        root.update_idletasks()

        messagebox.showinfo("Conversion Complete",
                            "PNG images have been converted back to DICOM format successfully.")

    except Exception as e:
        messagebox.showerror("Error", traceback.format_exc())

    finally:
        progress_bar.pack_forget()


def start_prediction_threaded():
    """Arayuz donmasin diye tahmini ayri thread'de calistirir."""
    threading.Thread(target=start_prediction).start()


def clear_folder_contents(folder):
    folder = str(folder)
    if not os.path.isdir(folder):
        return
    for item in os.listdir(folder):
        item_path = os.path.join(folder, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)


def restart():
    """Tum ara klasorleri temizleyerek baslangic durumuna doner."""
    try:
        for folder in (SOURCE_DIR, NORMAL_DIR, ABNORMAL_DIR, SEGMENTED_DIR,
                       OUTPUT_DIR, RESULT_DICOM_DIR):
            clear_folder_contents(folder)

        messagebox.showinfo("Restart", "All folders have been cleared.")
    except Exception as e:
        messagebox.showerror("Error", str(e))


# --------------------------------------------------------------------------
# DICOM goruntuleyici
# --------------------------------------------------------------------------
class DICOMViewer:
    def __init__(self, master, dicom_folder):
        self.master = master
        self.dicom_folder = dicom_folder
        self.dicom_files = self.load_and_sort_dicom_files(dicom_folder)
        self.current_index = 0
        self.original_image = None

        self.canvas = Canvas(master, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.photo_image = None
        self.update_image()

        master.bind("<MouseWheel>", self.on_mousewheel)
        master.bind("<Configure>", self.on_resize)

    def load_and_sort_dicom_files(self, folder):
        dicom_files = [os.path.join(folder, f) for f in os.listdir(folder)]
        dicom_files.sort(key=lambda x: pydicom.dcmread(x, stop_before_pixels=True).InstanceNumber)
        return dicom_files

    def load_dicom_image(self, dicom_path):
        ds = pydicom.dcmread(dicom_path)
        image_data = ds.pixel_array

        if ds.PhotometricInterpretation == "MONOCHROME1":
            image_data = np.max(image_data) - image_data

        image_data = apply_voi_lut(image_data, ds)

        if image_data.dtype != np.uint8:
            image_data = (np.divide(image_data, np.max(image_data)) * 255).astype(np.uint8)

        self.original_image = Image.fromarray(image_data).convert('RGB')

    def update_image(self):
        self.load_dicom_image(self.dicom_files[self.current_index])
        self.display_image()

    def display_image(self):
        window_width, window_height = self.canvas.winfo_width(), self.canvas.winfo_height()
        original_width, original_height = self.original_image.size
        ratio = min(window_width / original_width, window_height / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        resized_image = self.original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        self.canvas.delete("all")
        image_x = (window_width - new_width) / 2
        image_y = (window_height - new_height) / 2
        self.photo_image = ImageTk.PhotoImage(resized_image)
        self.canvas.create_image(image_x, image_y, anchor=tk.NW, image=self.photo_image)

    def on_resize(self, event):
        if self.original_image:
            self.display_image()

    def on_mousewheel(self, event):
        if event.delta > 0:
            self.prev_image()
        else:
            self.next_image()

    def next_image(self):
        if self.current_index < len(self.dicom_files) - 1:
            self.current_index += 1
            self.update_image()

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_image()


def launch_dicom_viewer():
    dicom_folder = str(RESULT_DICOM_DIR)
    if os.path.exists(dicom_folder):
        viewer_window = tk.Toplevel(root)
        viewer_window.title("LungScan DICOM Viewer")
        viewer_window.geometry("512x512")
        DICOMViewer(viewer_window, dicom_folder)
    else:
        messagebox.showinfo("DICOM Viewer", "The specified DICOM folder does not exist.")


# --------------------------------------------------------------------------
# Arayuz
# --------------------------------------------------------------------------
root = tk.Tk()
root.title("LungScan - DICOM Processor")

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
size = (320, 290)
x = int((screen_width / 2) - (size[0] / 2))
y = int((screen_height / 2) - (size[1] / 2))
root.resizable(False, False)
root.geometry(f'{size[0]}x{size[1]}+{x}+{y}')

start_index_label = tk.Label(root, text="Start Index:")
start_index_label.pack()
start_index_entry = tk.Entry(root)
start_index_entry.insert(0, str(DEFAULT_START_INDEX))
start_index_entry.pack()

end_index_label = tk.Label(root, text="End Index:")
end_index_label.pack()
end_index_entry = tk.Entry(root)
end_index_entry.insert(0, str(DEFAULT_END_INDEX))
end_index_entry.pack()

process_button = tk.Button(root, text="Start Process (DICOM -> PNG)", command=start_process)
process_button.pack()

start_prediction_button = tk.Button(root, text="Start Prediction",
                                    command=start_prediction_threaded)
start_prediction_button.pack()

view_image_button = tk.Button(root, text="View Result", command=launch_dicom_viewer)
view_image_button.pack()

restart_button = tk.Button(root, text="Restart", command=restart)
restart_button.pack()

progress_var = tk.DoubleVar()
progress_bar = Progressbar(root, variable=progress_var, maximum=100)
progress_bar.pack()


if __name__ == "__main__":
    root.mainloop()
