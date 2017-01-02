import numpy as np 
from mnist import MNIST
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from display_network import *



# def load_mnist_train():
# 	mndata = MNIST(MNIST_PATH)
# 	Train = mndata.load_training()
# 	## convert training data in to numpy array 
# 	X = np.array(Train[0])
# 	return X 

mndata = MNIST('../MNIST/')
mndata.load_training()
mndata.load_testing()
l = dir(mndata)
# X = mndata.train_images
X = mndata.test_images
X = np.asarray(X)

K = 10

# print type(X), X.shape
kmeans = KMeans(n_clusters=K).fit(X)

pred_label = kmeans.predict(X)

# display 10 data for each cluster 
X1 = np.zeros((40*K, 784))
id = 0 
for k in range(K):
	Xk = X[pred_label == k, :]
	X1[40*k: 40*k + 40,:] = Xk[:40,:]
	id += Xk.shape[0] 

# X2 = X.T 
A = display_network(X1.T)
plt.imshow(A, interpolation='nearest' )
plt.gray()
plt.show()


# print A.shape

# print 'done'
# plt.show(A)
