import numpy as np 
import matplotlib.pyplot as plt

# height (cm)
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]])
# weight (kg)
y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]])
# Visualize data 
# plt.plot(X, y, 'ro')
# plt.axis([140, 190, 45, 75])
# plt.xlabel('Height (cm)')
# plt.ylabel('Weight (kg)')
# plt.show()

# Building Xbar 
one = np.ones((1,X.shape[1]))
Xbar = np.concatenate((X, one), axis = 0)

# print Xbar, y

A = np.dot(Xbar, Xbar.T)
b = np.dot(Xbar, y.T)
w = np.dot(np.linalg.pinv(A), b)
print 'w = ', w


def cost(w):
	return .5/Xbar.shape[1]*np.linalg.norm(y.T - np.dot(Xbar.T, w), 2)**2;

def grad(w):
	# A = np.dot(Xbar.T, w) - y.T 
	# return np.dot(Xbar, A)
	return np.dot(Xbar, np.dot(Xbar.T, w) - y.T)/Xbar.shape[1]

print cost(w)

eps = 1e-4
def numerical_grad(w, cost):
	g = np.zeros_like(w)
	for i in range(len(w)):
		w_p = w.copy()
		w_n = w.copy()
		w_p[i] += eps 
		w_n[i] -= eps
		# print '2 cost', cost(w_p), cost(w_n)
		g[i] = (cost(w_p) - cost(w_n))/(2*eps)
	return g 

def check_grad(w, cost, grad):
	w = np.random.rand(w.shape[0], w.shape[1])
	grad1 = grad(w)
	grad2 = numerical_grad(w, cost)

	return True if np.linalg.norm(grad1 - grad2) < 1e-6 else False 




alpha = 1e-8
w = [np.array([[.5], [-30]])]

print check_grad(w[0], cost, grad)

print np.dot(Xbar, y.T)
# title = '$f(x) = x^2 + %dsin(x)$; ' %c_sin
# title += '$x_0 =  %.2f$; ' %x[0]
# title += r'$\alpha = %.2f$ ' % alpha 
# file_name = 'gd_14.gif'

for it in range(100000):
	# print cost(w[-1])
	w_new = w[-1] - alpha*grad(w[-1])

	if np.linalg.norm(w_new - w[-1]) < 1e-10:
		break 
	w.append(w_new)
	# print w[-1]
print w[-1]
print cost(w[-1])
# print it 
# for i in range(it):
# 	print w[it]

