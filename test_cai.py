from ultralytics import YOLO
import cv2
import os
import matplotlib.pyplot as plt

# Load model đã train xong
model = YOLO(r"F:\tailieu\HOC_MAY\code\model\yolo11_leaf_seg.pt")  # đổi path nếu cần

input_folder = r"F:\tailieu\HOC_MAY\code\dataset_split_leaf\images\test"   # folder chứa ảnh test
output_folder = r"F:\tailieu\HOC_MAY\code\predict_yolo11" # folder lưu ảnh kết quả

os.makedirs(output_folder, exist_ok=True)  # tạo folder output nếu chưa tồn tại

# Duyệt tất cả ảnh trong folder input
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Không đọc được ảnh: {img_path}")
            continue
        
        # Dự đoán segmentation
        results = model.predict(img, conf=0.25, save=False)
        result = results[0]

        # Lấy ảnh có vẽ mask, box, label
        img_result = result.plot()

        # Lưu ảnh kết quả vào folder output
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, img_result)

        print(f"Đã xử lý và lưu: {output_path}")
