import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Đọc dữ liệu
df = pd.read_excel('data_exel.xlsx', engine='openpyxl')

# Chọn 10 đặc trưng đầu vào và nhãn đầu ra
features = ['AGE', 'SEX', 'BMI', 'BP', 'TC', 'LDL', 'HDL', 'TCH', 'LTG', 'GLU']
X = df[features].values
y = df['Y'].values.reshape(-1, 1)

# Tính nghiệm hồi quy tuyến tính
w_star = np.linalg.inv(X.T @ X) @ X.T @ y

# Hàm tính loss
def eval(X, y, w):
    return np.square(X @ w - y).mean()

# In kết quả
print("Vector w* (trọng số):\n", w_star)
print("Optimal loss (MSE):", eval(X, y, w_star))

# Dự đoán đầu ra từ mô hình
y_pred = X @ w_star

# Vẽ biểu đồ so sánh y thực tế và y dự đoán


plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, s=10, alpha=0.7, label='Dự đoán')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Đường y = ŷ')
plt.xlabel("Giá trị thực tế (y)")
plt.ylabel("Giá trị dự đoán (ŷ)")
plt.title("So sánh giữa y và y dự đoán")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
