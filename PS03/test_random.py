import numpy as np
n = 4
a = np.arange(0, n, 1)
print(a)
np.random.shuffle(a)

a = np.array([1, 2, 2, 23])
x = np.random.choice(a)
b = np.where(a == 2)
print(b)
print(x)
print(a)