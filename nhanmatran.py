import numpy as np

# Khởi tạo ma trận X (4x5)
X = np.array([
    [1, 2104, 5, 1, 45],
    [1, 1416, 3, 2, 40],
    [1, 1534, 3, 2, 30],
    [1, 852, 2, 1, 36]
])

# Vector y (4x1)
y = np.array([
    [460],
    [232],
    [315],
    [178]
])

# Tính w = (X^T X)^(-1) X^T y
X_transpose = X.T
w = np.linalg.inv(X_transpose @ X) @ X_transpose @ y

# In kết quả
print("Vector w:")
print(w)
