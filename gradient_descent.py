import numpy as np

# Artificial sdata
true_w = np.array([1,2,3,4,5])
d = len(true_w)
points = []

for i in range(1000):
    x = np.random.randn(d)
    y = true_w.dot(x) + np.random.randn()
    points.append((x, y))

# Gradient descent

def F(w):
    return sum((w.dot(x) - y) ** 2 for x, y in points) / len(points)

def dF(w):
    return sum(2 * (w.dot(x) - y) * x for x, y in points) / len(points)


def gradientDescent(F, dF, d):
    w = np.zeros(d)
    eta = 0.01

    for t in range(1000):
        value = F(w)
        gradient = dF(w)
        w = w - eta * gradient
        print('iteration {}: w = {}, F(w) = {}'.format(t, w, value))

gradientDescent(F, dF, d)
