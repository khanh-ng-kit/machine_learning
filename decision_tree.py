import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder


#Đọc dữ liệu exel
filepath ="data_naive.xlsx"
data = pd.read_excel(filepath)

#Lấy dữ liệu đầu vào X, Y
X = data[['W1', 'W2', 'W3', 'W4', 'W5']]
y = LabelEncoder().fit_transform(data['Spam'])  # 'Yes' -> 1, 'No' -> 0
print("Đây là dữ liệu của y:")
print(y);

#Huấn luyện mô hình Decision Tree
clf = DecisionTreeClassifier(criterion='entropy', random_state=0)
clf.fit(X, y)

#Dự đoán gói tin mới D = {W1=1, W4=1, W5=1} (các W2 và W3 không có: đặt = 0)
new_packet = [[1, 0, 0, 1, 1]]
predicted_label = clf.predict(new_packet)[0]
label_mapping = {0: 'No', 1: 'Yes'}

#Hiển thị kết quả
print("Kết quả dự đoán:", label_mapping[predicted_label])

#In ra cây quyết định ở dạng văn bản
tree_rules = export_text(clf, feature_names=list(X.columns))
print("\nCây quyết định:")
print(tree_rules)