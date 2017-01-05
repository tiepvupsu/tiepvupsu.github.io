import math
import numpy as np 

import matplotlib.pyplot as plt

def grad(x):
	return 2*x+ 3*np.cos(3*x)

def cost(x):
	return x**2 + np.sin(3*x)

alpha = .1
x = [5]

for it in range(100):
	print cost(x[-1])
	x_new = x[-1] - alpha*grad(x[-1])

	if np.linalg.norm(x_new - x[-1]) < 1e-7:
		break 
	x.append(x_new)


x = np.asarray(x)
# print it, x_old
x0 = np.linspace(-5, 5, 1000)
y0 = cost(x0)

y = cost(x)
g = grad(x)
# plt.plot(x0, y0)
# plt.plot(x, y, 'ro', markersize=7)


fig, ax = plt.subplots()

def update(ii):
    label2 = 'iteration %d/%d: ' %(ii, it) + 'cost = %.2f' % y[ii] + ', grad = %.4f' %g[ii]

    animlist = plt.cla()
    # animlist = plt.axis('equal')
    animlist = plt.axis([-6, 6, -2, 30])

    animlist = plt.plot(x0, y0)

    if ii == 0:
    	animlist = plt.plot(x[ii], y[ii], 'ro', markersize = 7)
    else:
    	animlist = plt.plot(x[ii-1], y[ii-1], 'ko', markersize = 7)
    	animlist = plt.plot([x[ii-1], x[ii]], [y[ii-1], y[ii]], 'k-')
    	animlist = plt.plot(x[ii], y[ii], 'ro', markersize = 7)

    ax.set_xlabel(label2)
    return animlist, ax
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation 

anim = FuncAnimation(fig, update, frames=np.arange(0, it), interval=1000)

plt.show()

