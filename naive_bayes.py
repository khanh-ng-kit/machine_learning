import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import LabelEncoder

# Bước 1: Đọc dữ liệu từ file Excel
file_path = 'data_naive.xlsx'  # Đường dẫn đến file dữ liệu
data = pd.read_excel(file_path)

# Bước 2: Chuẩn bị dữ liệu
X = data[['W1', 'W2', 'W3', 'W4', 'W5']]
y = LabelEncoder().fit_transform(data['Spam'])  # 'Yes' -> 1, 'No' -> 0
print("Đây là dữ liệu của y:")
print(y);
# Bước 3: Huấn luyện mô hình Naive Bayes
model = BernoulliNB()
model.fit(X, y)

# Bước 4: Gói tin cần dự đoán
# Ví dụ: D = {W1=1, W4=1, W5=1} => W2 và W3 không có, ta cho là 0
new_packet = [[1, 0, 0, 1, 1]]

# Bước 5: Dự đoán
predicted_label = model.predict(new_packet)[0]
predicted_proba = model.predict_proba(new_packet)[0]

# Bước 6: Hiển thị kết quả
label_mapping = {0: 'No', 1: 'Yes'}
print("Kết quả dự đoán:", label_mapping[predicted_label])
print("Xác suất không phải spam (No):", round(predicted_proba[0] * 100, 2), "%")
print("Xác suất là spam (Yes):", round(predicted_proba[1] * 100, 2), "%")
