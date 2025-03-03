import numpy as np

np.set_printoptions(precision=3, suppress=True)

d = 10

np.random.seed(0)
points = np.linspace([-5, 0, 0], [5, 0, 0], 11) + np.random.normal(0, 0.01, (11, 3))
points[6:] += np.array([0, d, 0])

# center points around middle point
centered = points - points[5]
# remove it cuz it won't impact covariance (it's 0 now)
centered = np.delete(centered, 5, axis=0)

# compute the covariance
cov = centered.T @ centered
eigvals, eigvecs = np.linalg.eig(cov)
print(eigvals)
