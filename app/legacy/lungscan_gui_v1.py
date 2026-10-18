import tkinter as tk
from tkinter import messagebox
import os
from pathlib import Path
import shutil
import pydicom
from PIL import Image
from tkinter import messagebox, filedialog
import numpy as np
from keras.models import load_model
import traceback
import sys
import h5py
import tensorflow as tf
import os
import copy  # Eksik olan modülü içe aktar
import numpy as np
import pydicom
from PIL import Image
from pydicom.dataset import Dataset, FileMetaDataset
import cv2
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox, filedialog
import threading
import tkinter as tk
from tkinter import filedialog , Canvas
from PIL import Image, ImageTk
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np
from tkinter.ttk import Progressbar

# --------------------------------------------------------------------------
# YAPILANDIRMA (arsiv surumu icin sonradan eklenmistir)
# --------------------------------------------------------------------------
WORK_DIR = Path(os.environ.get("LUNGSCAN_WORK_DIR", Path(__file__).resolve().parent)).expanduser()
APP_DIR = Path(os.environ.get("LUNGSCAN_MODEL_DIR", WORK_DIR)).expanduser()







source_folder = None

# kaynak klasörü alıyor ve içini temizliyor
def clean_and_move_dicom_files(source_folder):
    
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








# kaynak klasörden(source) hedef klasörüne (destination) aktarıyor 
def dicom_to_png_filtered(source_folder, destination_folder , start_index, end_index):
    clean_and_move_dicom_files(source_folder)
    

    files = [os.path.join(source_folder, f) for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    
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

                # Save as PNG while preserving the original file name
                original_file_name = os.path.splitext(os.path.basename(file_path))[0] + ".png"
                image.save(os.path.join(destination_folder, original_file_name), "PNG")
            else:
                print(f"No PixelData found in: {file_path}")
        else:
            os.remove(file_path)

    









# hedef klasöründen (destination_folder) alıp predict atıyor 
def predict_and_move_images(model_path, destination_folder, target_normal_folder, target_abnormal_folder, progress_var):
    global source_folder
    global progress_bar

    try:
        print(f"Loading model from: {model_path}")
        model_path = 'model_in_5_1.h5'
        model = tf.keras.models.load_model(model_path)
        
        image_files = [f for f in os.listdir(destination_folder) if f.endswith('.png')]
        steps_per_image = 35 / len(image_files)  # Determine steps to increment per image

        for index, image_file in enumerate(image_files):
            image_path = os.path.join(destination_folder, image_file)
            img = Image.open(image_path).resize((256, 256))
            img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
            prediction = model.predict(img_array)
        
            if np.argmax(prediction) == 0:  # Assuming 0 is abnormal
                target_folder = target_abnormal_folder
            else:
                target_folder = target_normal_folder
        
            os.makedirs(target_folder, exist_ok=True)
            os.replace(image_path, os.path.join(target_folder, image_file))

            progress_var.set(5 + (index + 1) * steps_per_image)
            root.update_idletasks()
            
    except Exception as e:
        messagebox.showerror("Error", str(e))
        print(str(e))





# yeni dicom verilerini oluşturuyor 
        
def is_color_dicom(dicom_path):
    # DICOM dosyasını oku
    ds = pydicom.dcmread(dicom_path)
    # Photometric Interpretation değerine bakarak renkli olup olmadığını kontrol et
    return ds.PhotometricInterpretation.startswith('RGB')

def convert_to_color(png_image):
    # Siyah beyaz (1 kanal) PNG görüntüyü renkli (3 kanal) bir görüntüye çevir
    return png_image.convert('RGB')

def resize_image(png_image, width, height):
    # Görüntüyü istenilen boyutlara uyacak şekilde yeniden boyutlandır
    return png_image.resize((width, height))

def create_dicom_from_png(png_image, original_dicom, instance_number):
    # Orijinal DICOM verilerini kopyala
    new_dicom = copy.deepcopy(original_dicom)

    # PNG görüntüsünü Numpy dizisine dönüştür ve PixelData olarak ayarla
    new_dicom.PixelData = np.array(png_image).tobytes()

    # Görüntünün boyutlarını güncelle
    new_dicom.Rows, new_dicom.Columns = png_image.size

    # Eğer orijinal DICOM renkli değilse, Photometric Interpretation'ı güncelle
    if not is_color_dicom(original_dicom.filename):
        new_dicom.PhotometricInterpretation = 'RGB'

    # Örneklerin piksel başına sayısını güncelle
    new_dicom.SamplesPerPixel = 3
    new_dicom.BitsAllocated = 8
    new_dicom.BitsStored = 8
    new_dicom.HighBit = 7
    new_dicom.PixelRepresentation = 0

    # Transfer Syntax ve diğer meta verileri ayarla
    new_dicom.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    # Instance Number'ı güncelle
    new_dicom.InstanceNumber = instance_number

    return new_dicom





# png i dicom a çeviriyor 
def convert_png_to_dicom(png_folder, original_dicom_folder, output_folder):
    # PNG ve DICOM dosya yollarını kontrol et
    instance_numbers = {}  # Sıralama numarasını tutmak için sözlük

    # Dosyaların instance numarasını al ve bir listeye kaydet
    for dicom_filename in os.listdir(original_dicom_folder):
        dicom_path = os.path.join(original_dicom_folder, dicom_filename)
        original_dicom = pydicom.dcmread(dicom_path)
        instance_number = original_dicom.get('InstanceNumber', 0)
        instance_numbers[dicom_filename] = instance_number

    # Sıralama numarasına göre PNG dosyalarını sırala
    sorted_png_filenames = sorted(os.listdir(png_folder), key=lambda fname: instance_numbers.get(fname.replace('.png', ''), 0))

    # Sıralanmış dosyalar üzerinde işlem yap
    for i, png_filename in enumerate(sorted_png_filenames, start=1):
        if png_filename.lower().endswith('.png'):
            # Yolları tanımla
            png_path = os.path.join(png_folder, png_filename)
            dicom_filename_without_ext = png_filename.split('.')[0]
            original_dicom_path = os.path.join(original_dicom_folder, dicom_filename_without_ext)
            output_dicom_path = os.path.join(output_folder, dicom_filename_without_ext )

            # Orijinal DICOM dosyasını yükle
            original_dicom = pydicom.dcmread(original_dicom_path)

            # PNG görüntüsünü yükle
            png_image = Image.open(png_path).convert('RGB')

            # Renk ve boyut ayarlamalarını yap
            png_image = resize_image(png_image, 512, 512)

            # Yeni DICOM oluştur ve InstanceNumber ayarla
            new_dicom = create_dicom_from_png(png_image, original_dicom, i)

            # Yeni DICOM dosyasını kaydet
            new_dicom.save_as(output_dicom_path)

            print(f"Converted {png_filename} to DICOM and saved as {output_dicom_path} with InstanceNumber {i}")






def start_process():
    global source_folder
    try:
        # kaynak klasörü kullanıcı dan alıyor
        source_folder = str(WORK_DIR / "kaynak_klasor")
        if not source_folder:  # Check if the user canceled the dialog
            return

        # Set the destination folder path
        destination_folder = str(WORK_DIR / "hedef_klas")

        # Convert the start and end indexes from user input
        start_index = int(start_index_entry.get())
        end_index = int(end_index_entry.get())

        # Process the DICOM files
        dicom_to_png_filtered(source_folder, destination_folder, start_index, end_index)

        messagebox.showinfo("Success", "Process completed successfully.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Make sure the rest of your GUI setup code is here
        prediction_button = tk.Button(root, text="Start Prediction", command=start_prediction)
        prediction_button.pack()




class DICOMViewer:
    def __init__(self, master, dicom_folder):
        self.master = master
        self.dicom_folder = dicom_folder
        self.dicom_files = self.load_and_sort_dicom_files(dicom_folder)
        self.current_index = 0
        self.original_image = None

        self.canvas = Canvas(master, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.photo_image = None  # Görüntü için bir placeholder

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

        self.canvas.delete("all")  # Önceki çizimleri temizle
        image_x = (window_width - new_width) / 2
        image_y = (window_height - new_height) / 2
        self.photo_image = ImageTk.PhotoImage(resized_image)
        self.canvas.create_image(image_x, image_y, anchor=tk.NW, image=self.photo_image)

    def on_resize(self, event):
        if self.original_image:
            self.display_image()

    def on_mousewheel(self, event):
        if event.delta > 0:  # Yukarı scroll
            self.prev_image()
        else:  # Aşağı scroll
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
    dicom_folder = str(WORK_DIR / "hasta_sonuclari")
    if os.path.exists(dicom_folder):
        # Create a new Toplevel window for the DICOM viewer
        viewer_window = tk.Toplevel(root)
        viewer_window.title("DICOM Viewer")
        viewer_window.geometry("512x512") 
        DICOMViewer(viewer_window, dicom_folder)
    else:
        messagebox.showinfo("DICOM Viewer", "The specified DICOM folder does not exist.")




def start_prediction():
    global source_folder
    global progress_var, progress_bar


    # Reset the progress bar
    progress_var.set(0)
    root.update_idletasks()

    try:
        # Calculate total progress steps
        model_path = str(APP_DIR / "model_in_5_1.h5")
        destination_folder = str(WORK_DIR / "hedef_klas")
        target_normal_folder = str(WORK_DIR / "normal")
        target_abnormal_folder = str(WORK_DIR / "abnormal")
        
        # Load the classification model
        tf.keras.models.load_model(model_path)
        progress_var.set(5)  # Assume model loading is 10% of the work
        root.update_idletasks()

        # Classification and moving images
        predict_and_move_images(model_path, destination_folder, target_normal_folder, target_abnormal_folder, progress_var)
        

        # Load the segmentation model
        yolo_model = YOLO(str(APP_DIR / "best.pt"))
        segmented_dir = str(WORK_DIR / "segmented")
        os.makedirs(segmented_dir, exist_ok=True)
        predict_images_dir = target_abnormal_folder

        # Process each image for segmentation
        image_files = os.listdir(predict_images_dir)
        per_image_increment = 50 / len(image_files)  # Assuming segmentation is 70% of the work
        
        for index, image_file in enumerate(image_files):
            if image_file.endswith('.png'):
                img_path = os.path.join(predict_images_dir, image_file)
                img = cv2.imread(img_path)
                H, W, _ = img.shape
                
                results = yolo_model(img)
                progress_var.set(40 + (index + 1) * per_image_increment)
                root.update_idletasks()
            
                
                if results[0].masks is not None:
                    predicted_masks = results[0].masks.data.cpu().numpy()
                    for mask in predicted_masks:
                        mask = cv2.resize(mask, (W, H))
                        _, mask_thresh = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
                        mask_thresh = mask_thresh.astype(np.uint8)
                        contours, _ = cv2.findContours(mask_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(img, contours, -1, (0, 255, 0), 0)
                
                save_path = os.path.join(segmented_dir, image_file)
                cv2.imwrite(save_path, img)





        normal_dir = str(WORK_DIR / "normal")
        segmented_dir = str(WORK_DIR / "segmented")
        output_dir = str(WORK_DIR / "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Copy images from 'normal' directory
        for filename in os.listdir(normal_dir):
            if filename.endswith('.png'):
                src_path = os.path.join(normal_dir, filename)
                dst_path = os.path.join(output_dir, filename)
                shutil.copy(src_path, dst_path)

        # Copy images from 'segmented' directory
        for filename in os.listdir(segmented_dir):
            if filename.endswith('.png'):
                src_path = os.path.join(segmented_dir, filename)
                dst_path = os.path.join(output_dir, filename)
                shutil.copy(src_path, dst_path)

        print("All images have been copied to the 'output' folder.")
        progress_var.set(95)  # Copying images is another 5%
        root.update_idletasks()
        

    except Exception as e:
        messagebox.showerror("Error", traceback.format_exc())
     # Inside your start_prediction function, after the YOLO segmentation part
        


    try:
    
    # Paths for the PNG to DICOM conversion
        png_folder = str(WORK_DIR / "output")
        original_dicom_folder = source_folder  # Adjust as needed
        output_folder = str(WORK_DIR / "hasta_sonuclari")
        os.makedirs(output_folder, exist_ok=True)

    # Convert PNGs back to DICOM format
        convert_png_to_dicom(png_folder, original_dicom_folder, output_folder)

        progress_var.set(100)  # Completion of DICOM conversion is the last step
        root.update_idletasks()

        messagebox.showinfo("Conversion Complete", "PNG images have been converted back to DICOM format successfully.")
      
    except Exception as e:
        messagebox.showerror("Error", traceback.format_exc())  # Use traceback for more detailed error information

    finally:
        # Hide the progress bar after completion or error
        progress_bar.pack_forget()
       
    
        
def clear_folder_contents(folder):
    for item in os.listdir(folder):
        item_path = os.path.join(folder, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

def restart():
    global source_folder
    try:
        folders_to_clear = [
            str(WORK_DIR / "kaynak_klasor"),
            str(WORK_DIR / "normal"),
            str(WORK_DIR / "abnormal"),
            str(WORK_DIR / "segmented"),
            str(WORK_DIR / "output"),
            str(WORK_DIR / "hasta_sonuclari")
            
        ]
        
        for folder in folders_to_clear:
            clear_folder_contents(folder)



        messagebox.showinfo("Restart", "All folders have been cleared.")
    except Exception as e:
        messagebox.showerror("Error", str(e))



# GUI setup
root = tk.Tk()
root.title("DICOM Processor")

# Calculate the center position
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
size = (300, 250)
x = int((screen_width / 2) - (size[0] / 2))
y = int((screen_height / 2) - (size[1] / 2))
root.resizable(False, False)  # This prevents the window from being resized


# Set the window size and position it at the center
root.geometry(f'{size[0]}x{size[1]}+{x}+{y}')

start_index_label = tk.Label(root, text="Start Index:")
start_index_label.pack()
start_index_entry = tk.Entry(root)
start_index_entry.pack()

end_index_label = tk.Label(root, text="End Index:")
end_index_label.pack()
end_index_entry = tk.Entry(root)
end_index_entry.pack()

process_button = tk.Button(root, text="Start Process", command=start_process)
process_button.pack()

prediction_button = tk.Button(root, text="Start Prediction", command=start_prediction)
prediction_button.pack()

view_image_button = tk.Button(root, text="Viewing Image", command=launch_dicom_viewer)
view_image_button.pack()


restart_button = tk.Button(root, text="Restart", command=restart)
restart_button.pack()

progress_var = tk.DoubleVar()
progress_bar = Progressbar(root, variable=progress_var, maximum=100)
progress_bar.pack()


root.mainloop()