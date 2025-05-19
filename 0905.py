import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Tạo dữ liệu từ bảng
data = {
    'A': [1, 1],
    'B': [0, 1],
    'C': [0, 6],
    'D': [3, 6],
    'E': [2, 1],
    'F': [1, 2],
    'G': [5, 2],
    'H': [4, 5],
    'I': [3, 4],
    'J': [3, 5],
    'K': [2, 4],
    'L': [2, 6],
}
points = np.array(list(data.values())) # Chuyển thành mảng numpy (shape 12x2)
labels = list(data.keys()) # Lưu tên điểm để hiển thị sau

#a CHạy thuật toán K-means với tâm cụm C1=(0,0) và C2=(2,2). Số vòng lặp là 3

def manhattan_kmeans(X, k, centroids, iterations=3):
    centroids = np.array(centroids)
    for _ in range(iterations):
        distances = cdist(X, centroids, metric='cityblock') # khoảng cách L1
        clusters = np.argmin(distances, axis=1) #gán điểm về cụm gần nhất 
        for i in range(k):
            if np.any(clusters == i):
                centroids[i] = np.median(X[clusters == i], axis=0) #cập nhật tâm cụm bằng trung vị
    print("Đây là trung vị:")
    print(centroids)
    print("Còn đây là khoảng cách L1:")
    print(distances)    
    return clusters, centroids

# Chạy với C1 = (0, 0), C2 = (2, 2), 3 vòng lặp
init_centroids = [(0, 0), (2, 2)]
kmeans_clusters, final_centroids = manhattan_kmeans(points, 2, init_centroids)

print("KMeans (L1 distance) labels:", kmeans_clusters)

#b DBSCAN sử dụng khoảng cách L1
dbscan = DBSCAN(eps=1.95, min_samples=3, metric='cityblock') 
#eps bán kính vùng lân cận, min_samples là số điểm tối thiểu để 1 điểm là core

db_labels = dbscan.fit_predict(points)

# Phân loại
core_samples_mask = np.zeros_like(db_labels, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True

for i, label in enumerate(db_labels):
    label_type = "Noise"
    if label != -1:
        if core_samples_mask[i]:
            label_type = f"Core (Cluster {label})"
        else:
            label_type = f"Border (Cluster {label})"
    print(f"Point {labels[i]}: {points[i]} -> {label_type}")

plt.figure(figsize=(8, 6))
unique_labels = set(db_labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    class_member_mask = (db_labels == k)
    xy = points[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', label=f"Cluster {k}" if k != -1 else "Noise")
    
    xy = points[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'x', markerfacecolor=tuple(col), markeredgecolor='k')

for i, label in enumerate(labels):
    plt.text(points[i][0]+0.1, points[i][1]+0.1, label)

plt.title("DBSCAN clustering")
plt.legend()
plt.grid(True)
plt.show() 