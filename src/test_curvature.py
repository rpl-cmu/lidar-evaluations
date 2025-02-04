# import sympy as sy
# from sympy import pprint as print
import numpy as np

# ------------------------- Symbolic ------------------------- #
# r = sy.Symbol("r")
# n = 1

# gpp = sy.ZeroMatrix(2, 1)
# gp = sy.ZeroMatrix(2, 1)
# p = sy.Matrix([r * sy.cos(0), r * sy.sin(0)])
# for i in range(n):
#     theta = sy.Symbol(f"theta_{i}")
#     p_m1 = sy.Matrix([r * sy.cos(-theta), r * sy.sin(-theta)])
#     p_1 = sy.Matrix([r * sy.cos(theta), r * sy.sin(theta)])

#     # gpp += (p_m1 - p) + (p_1 - p)
#     # gp += (p_m1 - p) + (p - p_1)

#     gp += (p_1 - p_m1) / 2
#     gpp += (p_1 - 2 * p + p_m1) / 2

# # gp /= 2 * n
# # gpp /= 2 * n

# print(gp)
# print(gpp)

# ------------------------- Vector approx ------------------------- #
# kappa = sy.sqrt((gp[0] * gpp[1] - gp[1] * gpp[0]) ** 2 / (gp[0] ** 2 + gp[1] ** 2) ** 3)
# print(kappa.expand().simplify())

# n = 1000
# r = 3
# gp = np.zeros(2)
# gpp = np.zeros(2)
# p = np.array([r * np.cos(0), r * np.sin(0)])
# for i in np.linspace(0.00001, 0.1, n):
#     p_m1 = np.array([r * np.cos(-i), r * np.sin(-i)])
#     p_1 = np.array([r * np.cos(i), r * np.sin(i)])

#     gp += (p_1 - p_m1) / (2 * np.linalg.norm((p_1 - p)))
#     gpp += (p_1 - 2 * p + p_m1) / (2 * np.linalg.norm((p_1 - p)))

# gp /= n
# gpp /= n

# kappa = (gp[0] * gpp[1] - gp[1] * gpp[0]) / (gp[0] ** 2 + gp[1] ** 2) ** (3 / 2)
# print(f"{gp=}")
# print(f"{gpp=}")
# print(f"{1/kappa=}")

# ------------------------- Circle compute ------------------------- #
r = 3
theta = np.pi / 6

p_m1 = np.array([r * np.cos(-theta), r * np.sin(-theta)])
p = np.array([r * np.cos(0), r * np.sin(0)])
p_1 = np.array([r * np.cos(theta), r * np.sin(theta)])

p_m1 = np.array([r, 1])
p_1 = np.array([r, -1])

a = np.linalg.norm(p - p_m1)
b = np.linalg.norm(p_1 - p)
c = np.linalg.norm(p_1 - p_m1)

s = (a + b + c) / 2
A = np.sqrt(s * (s - a) * (s - b) * (s - c))

R = a * b * c / (4 * A)
print(f"{R=}")

kappa = (4 * A) / (a * b * c)
print(f"{kappa=}")
