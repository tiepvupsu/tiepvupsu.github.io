# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib.pyplot as plt

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Building Xbar 
one = np.ones((X.shape[0],1))
Xbar = np.concatenate((one, X), axis = 1)

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w_exact = np.dot(np.linalg.pinv(A), b)


def cost(w):
	# return .5/Xbar.shape[0]*np.linalg.norm(y - Xbar.dot(w), 2)**2;
	return .5/Xbar.shape[0]*np.linalg.norm(y - Xbar.dot(w), 2)**2;

# print('The exact solution w = ', w, '; cost = %.5f'% cost(w))


def grad(w):
	return 1/Xbar.shape[0] * Xbar.T.dot(Xbar.dot(w) - y)


def numerical_grad(w, cost):
	eps = 1e-4
	g = np.zeros_like(w)
	for i in range(len(w)):
		w_p = w.copy()
		w_n = w.copy()
		w_p[i] += eps 
		w_n[i] -= eps
		g[i] = (cost(w_p) - cost(w_n))/(2*eps)
	return g 

def check_grad(w, cost, grad):
	w = np.random.rand(w.shape[0], w.shape[1])
	grad1 = grad(w)
	grad2 = numerical_grad(w, cost)
	return True if np.linalg.norm(grad1 - grad2) < 1e-6 else False 


print( 'Checking gradient...', check_grad(np.random.rand(2, 1), cost, grad))


def myGD(w_init, grad, eta):
	w = [w_init]
	for it in range(1000):
		w_new = w[-1] - eta*grad(w[-1])

		if np.linalg.norm(w_new - w[-1]) < 1e-5:
			break 
		w.append(w_new)
		# print('iter %d: ' % it, w[-1].T)
	return (w, it) 

w_init = np.random.randn(2, 1)
(w1, it1) = myGD(w_init, grad, 0.05)
(w2, it2) = myGD(w_init, grad, 0.5)
(w3, it3) = myGD(w_init, grad, 1)

print(it1, it2, it3)
# print('Exact solution      : ', w_exact.T, '; cost: ', cost(w_exact))
# print('Solution found by GD: ', w[-1].T, '; cost: ', cost(w[-1]))

