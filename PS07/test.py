import numpy as np
probs = np.array([[0.5], [0.5]])

for i in range(10):
    a_1 = np.random.choice(2, p=probs.ravel())
    print(a_1)