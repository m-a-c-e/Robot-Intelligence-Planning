import numpy as np

a = np.array([[1, 2, 3], [1, 2, 3]])
b = np.array([[1, 2, 3], [1, 2, 3]])
c = np.array(np.dot(a, b.T))
print(c)
print(np.matmul(a, b.T))