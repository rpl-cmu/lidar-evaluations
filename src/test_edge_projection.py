import numpy as np

# a = np.array([1, 2, 3])
# b = np.array([4, 5, 6])
# p = np.array([11, 8, 9])
np.random.seed(0)
a = np.random.rand(3)
b = np.random.rand(3)
p = np.random.rand(3)


# normal projection
out = np.linalg.norm(np.cross(p - a, p - b)) / np.linalg.norm(b - a)
print(out)

# my projection
d = (b - a) / np.linalg.norm(b - a)
A = np.eye(3) - np.outer(d, d)
v = (np.eye(3) - np.outer(d, d)).dot(p - b)
v = p - b
out = np.sqrt(v.T @ A @ v)
print(out)

# A = np.eye(3) - np.outer(d, d)
# print(A)
# print(A @ A)
# print(np.allclose(A @ A, A))

# P = np.outer(d, d)
# print(P)
# print(P @ P)
# print(np.allclose(P @ P, P))
