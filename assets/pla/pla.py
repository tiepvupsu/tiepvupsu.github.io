import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(16)

means = [[2, 2], [5, 3]]
cov = [[1, 0], [0, 1]]
N = 5
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)

plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 8, alpha = .8)
plt.plot(X1[:, 0], X1[:, 1], 'ro', markersize = 8, alpha = .8)
plt.axis('equal')
plt.show()

X = np.concatenate((X0, X1), axis = 0)

original_label =y = np.asarray([-1]*N + [1]*N).T

# Building Xbar 
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1).T 

# print(Xbar)
# print(y)


def h(w, x):
    # import pdb; pdb.set_trace()  # breakpoint 49457684 //
    return np.sign(np.dot(x.T, w))
from math import *
print np.sign(-1)
epoches = 1000

def has_converged(w, Xbar, y):
    yhat = h(Xbar, w)
    return np.linalg.norm(y - yhat) == 0

def pla(Xbar, y):
    w_init = np.random.randn(3, 1)
    w = [w_init]
    N = Xbar.shape[1]
    for epoch in range(epoches):
        for i in range(N):
            xi = Xbar[:, i].reshape(3, 1)
            yhat = h(w[-1], xi)[0][0]
            if yhat != y[i]:
                # import pdb; pdb.set_trace()  # breakpoint 385a096e //
                w_new = w[-1] + (y[i] - yhat)*xi
                w.append(w_new)
        if has_converged(w[-1], Xbar, y):
            return w 
    return w 

w = pla(Xbar, y)

# print(w)
print( h(w[-1], Xbar))            
print(len(w))

def draw_line(w):
    w0, w1, w2 = w[0], w[1], w[2]
    if w2 != 0:
        x11, x12 = 0, 10
        return plt.plot([x11, x12], [-(w1*x11 + w0)/w2, -(w1*x12 + w0)/w2])
    else:
        x10 = -w0/w1
        return plt.plot([x10, x10], [0, 10])

    plt.cla()
# draw_line([1, 2, 1])
# plt.show()
    

## GD example
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation 
def viz_alg_1d_2(w):
    it = len(w)    
       
    fig, ax = plt.subplots(figsize=(4, 4))  
    
    def update(i):
        ani = plt.cla()
        #points
        ani = plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 4, alpha = .8)
        ani = plt.plot(X1[:, 0], X1[:, 1], 'ro', markersize = 4, alpha = .8)
        ani = plt.axis([0 , 6, 0, 6])
        ani = draw_line(w[ i if i < it else it-1 ])
        label = 'GD without Momemtum: iter %d/%d' %(i, it)
        ax.set_xlabel(label)
        return ani, ax 
        
    anim = FuncAnimation(fig, update, frames=np.arange(0, it + 10), interval=500)
    anim.save('haha.gif', dpi = 100, writer = 'imagemagick')
    plt.show()
    
# x = np.asarray(x)
viz_alg_1d_2(w)