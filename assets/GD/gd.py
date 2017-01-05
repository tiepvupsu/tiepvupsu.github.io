import math
import numpy as np 

import matplotlib.pyplot as plt

def grad(x):
	return 2*x + math.cos(x)

def cost(x):
	return x**2 + np.sin(x)

alpha = .3
x = [6]
for it in range(100):
	print cost(x[-1])
	x_new = x[-1] - alpha*grad(x[-1])

	if np.linalg.norm(x_new - x[-1]) < 1e-8:
		break 
	x.append(x_new)

x = np.asarray(x)
# print it, x_old
x0 = np.linspace(-5, 5, 1000)
y0 = cost(x0)

y = cost(x)
plt.plot(x0, y0)
plt.plot(x, y, 'ro', markersize=7)

plt.show()

