import cv2
import numpy as np
import os
import shutil

from sklearn.model_selection import train_test_split

# Đường dẫn
image_dir = r"F:\tailieu\HOC_MAY\code\dataset\images"
label_dir = r"F:\tailieu\HOC_MAY\code\dataset\labels"
output_dir = "dataset_split_leaf"

# Tạo thư mục đầu ra
os.makedirs(f"{output_dir}/images/train", exist_ok=True)
os.makedirs(f"{output_dir}/images/val", exist_ok=True)
os.makedirs(f"{output_dir}/images/test", exist_ok=True)
os.makedirs(f"{output_dir}/labels/train", exist_ok=True)
os.makedirs(f"{output_dir}/labels/val", exist_ok=True)
os.makedirs(f"{output_dir}/labels/test", exist_ok=True)

# Lấy danh sách file ảnh (bỏ đuôi .jpg để map với label)
image_files = [f.replace(".jpg", "") for f in os.listdir(image_dir) if f.endswith(".jpg")]

# Chia train/test/val (tỷ lệ 70/15/15)
total_samples = len(image_files)
test_size = int(0.15 * total_samples)  # 15% test
val_size = int(0.15 * total_samples)   # 15% val

# Lần 1: Tách test (15%)
train_val, test = train_test_split(
    image_files, 
    test_size=test_size,
    random_state=42
)

# Lần 2: Tách val từ train_val (15% tổng số)
train, val = train_test_split(
    train_val,
    test_size=val_size,  # Số lượng chính xác thay vì tỷ lệ
    random_state=42
)
# Copy ảnh và labels vào thư mục tương ứng
def copy_files(files, split_type):
    for file in files:
        shutil.copy(f"{image_dir}/{file}.jpg", f"{output_dir}/images/{split_type}/{file}.jpg")
        shutil.copy(f"{label_dir}/{file}.txt", f"{output_dir}/labels/{split_type}/{file}.txt")

copy_files(train, "train")
copy_files(val, "val")
copy_files(test, "test")

shutil.make_archive('rice_data_rbf_split', 'zip', output_dir)
print("Da nen du lieu thanh file rice_data_rbf_split.zip, ban co the tai xuong file nay")

# Đếm số file trong mỗi tập
print("Số ảnh train:", len(os.listdir(f"{output_dir}/images/train")))
print("Số ảnh val:", len(os.listdir(f"{output_dir}/images/val")))
print("Số ảnh test:", len(os.listdir(f"{output_dir}/images/test")))

