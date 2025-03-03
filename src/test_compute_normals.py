import numpy as np

data = np.array([[-2, -0.2], [-1, 0.2], [0, 0], [1, 0], [2, 0.4]])

for i in range(1, 4):
    print(data[i - 1] - 2 * data[i] + data[i + 1])
