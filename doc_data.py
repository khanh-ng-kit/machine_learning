#@title The data {vertical-output: true}
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# _X = np.random.randn(1000, 1)
# X = np.concatenate([np.ones_like(_X), _X], axis=-1) # add bias
# print(X.shape, _X.shape)
# y = _X * 2 + 1 + np.random.randn(*_X.shape)

# Load data from an Excel file
df = pd.read_excel('data_exel.xlsx', engine='openpyxl')  # Read the Excel file
_X = df['TCH'].values.reshape(-1, 1)
X = np.concatenate([np.ones_like(_X), _X], axis=-1) # add bias

y = df['Y'].values.reshape(-1, 1)
# y2 = df['Trung bình']  # Second dataset


plt.scatter(_X, y, s=1)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Dataset')
# plt.show()
print(_X.shape, y.shape)

#@title Some helper functions
def eval(X, y, w):
    return np.square(X @ w - y).mean()

def plot(_X, y, w):
    plt.scatter(_X,y, s=1)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('TCH')
    print(_X)
    x_test = np.linspace(0, np.max(_X), 100)[..., None]
    x_test_pad = np.concatenate([np.ones_like(x_test), x_test], axis=-1)
    y_test = x_test_pad @ w
    plt.plot(x_test, y_test, c='red')
    plt.show()
    
#@title Analytical solution {vertical-output: true}
w_star = np.linalg.inv(X.T @ X) @ X.T @ y
print(w_star) 
plot(_X, y, w_star)
print("optimal loss:", eval(X, y, w_star))