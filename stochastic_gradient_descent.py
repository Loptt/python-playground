import numpy as np

# Artificial data
true_w = np.array([1,2,3,4,5])
d = len(true_w)
points = []

for i in range(1000):
    x = np.random.randn(d)
    y = true_w.dot(x) + np.random.randn()
    points.append((x, y))

# Gradient descent

def F(w, i):
    x, y = points[i]
    return (w.dot(x) - y) ** 2

def dF(w, i):
    x, y = points[i]
    return 2 * (w.dot(x) - y) * x


def gradientDescent(F, dF, d, n):
    w = np.zeros(d)
    eta = 0.01
    numUpdates = 0

    for t in range(1000):
        for i in range(n):
            value = F(w, i)
            gradient = dF(w, i)
            numUpdates += 1
            eta = 1.0 / numUpdates
            w = w - eta * gradient
        print('iteration {}: w = {}, F(w) = {}'.format(t, w, value))

gradientDescent(F, dF, d, len(points))
