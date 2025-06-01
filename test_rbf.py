import numpy as np
import cv2
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. XÂY DỰNG LỚP RBF TÙY CHỈNH
class RBFLayer(Layer):
    def __init__(self, centers, beta, **kwargs): #Hàm khởi tạo lớp
        super(RBFLayer, self).__init__(**kwargs)     #gọi hàm khởi tạo của lớp Layer, tích hợp vào Keras
        self.centers_np = centers if isinstance(centers, np.ndarray) else np.array(centers)  
        self.centers = tf.constant(self.centers_np, dtype=tf.float32)
        self.beta = beta

    def call(self, inputs): #Hàm chính nhận đầu vào và trả về đầu ra lớp RBF
        inputs = tf.reshape(inputs, [tf.shape(inputs)[0], -1]) #chuyển thành ma trận 2D [batch_size, số_feature];tf.shape(inputs)[0]:lấy kích thước batch
        C = tf.expand_dims(self.centers, 0) #self.centers có shape là [num_centers, features]; tf.expand_dims: thêm chiều mới ở vị trí 0, tạo thành[1, num_centers, features]
        X = tf.expand_dims(inputs, 1) #inputs có shape[batch_size, features];thêm chiều mới ở vị trí 1, tạo thành [batch_size, 1, features]
        dist = tf.reduce_sum((X - C) ** 2, axis=-1) #X-C: mỗi điểm trong batch sẽ trừ với center tương ứng; tính tổng bình phương theo chiều features
        return tf.exp(-self.beta * dist) #Tính hàm RBF Gaussian: exp(-beta.|x-c|^2)
    def get_config(self):
        config = super().get_config()
        config.update({
            "centers": self.centers_np.tolist(),  # cần chuyển về list
            "beta": self.beta,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

# Đường dẫn file model
model_path = r'F:\tailieu\HOC_MAY\code\model\rice_leaf_rbf_model.h5'
print("Tồn tại file model:", os.path.exists(model_path))

# Đường dẫn file class_indices
class_indices_path = r'F:\tailieu\HOC_MAY\code\output\output_rbf\class_indices.json'
if os.path.exists(class_indices_path):
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
    inv_class_indices = {int(v): k for k, v in class_indices.items()}
    class_names = [inv_class_indices[i] for i in range(len(inv_class_indices))]
    print("Đọc class_names từ class_indices.json:")
    print(class_names)
else:
    print("Không tìm thấy class_indices.json, dùng mặc định (có thể sai thứ tự).")
    class_names = ['Bacterialblight', 'Blast', 'Brownspot', 'Tungro']

# Load mô hình
model = load_model(model_path, custom_objects={"RBFLayer": RBFLayer})

# Tạo ImageDataGenerator cho tập test
test_dir = r'F:\tailieu\HOC_MAY\code\rice_data_rbf_split\test'
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Dự đoán toàn bộ tập test
test_generator.reset()
pred = model.predict(test_generator, verbose=1)
y_pred = np.argmax(pred, axis=1)
y_true = test_generator.classes

# Sắp xếp lại tên lớp theo thứ tự index để in báo cáo đúng
class_names_ordered = [None] * len(test_generator.class_indices)
for k, v in test_generator.class_indices.items():
    class_names_ordered[v] = k

# In báo cáo đánh giá
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names_ordered))

# Hiển thị các ảnh dự đoán sai
file_paths = test_generator.filepaths
wrong_indices = np.where(y_pred != y_true)[0]
print(f"\nSố ảnh dự đoán sai: {len(wrong_indices)}")

for i in wrong_indices:
    img_path = file_paths[i]
    true_label = class_names[y_true[i]] if y_true[i] < len(class_names) else "Unknown"
    pred_label = class_names[y_pred[i]] if y_pred[i] < len(class_names) else "Unknown"

    # Đọc ảnh
    img = cv2.imread(img_path)
    if img is None:
        print(f" Ảnh không tồn tại hoặc lỗi khi đọc: {img_path}")
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (64,64))
    img_array = img_resized / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Dự đoán lại để lấy độ tin cậy
    prob = model.predict(img_array, verbose=0)[0]
    conf = np.max(prob)

    # Hiển thị ảnh
    plt.figure(figsize=(4, 4))
    plt.imshow(img_rgb)
    plt.title(f"True: {true_label}\nPred: {pred_label}\nConf: {conf:.2f}")
    plt.axis('off')
    plt.show()
