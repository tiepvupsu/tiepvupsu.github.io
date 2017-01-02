# %reset
import numpy as np 
from mnist import MNIST
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

from display_network import *

mndata = MNIST('../MNIST/')

is_train_set = False 
# is_train_set = True 
if is_train_set:
	mndata.load_training()
	X = mndata.train_images
else:
	mndata.load_testing()
	X = mndata.test_images
X = np.asarray(X)

K = 10
if is_train_set:
	kmeans = MiniBatchKMeans(n_clusters=K, n_init=50, batch_size=20000).fit(X)
else:
	kmeans = MiniBatchKMeans(n_clusters=K, init='k-means++', n_init=10000, batch_size=5000).fit(X)
	# kmeans = KMeans(n_clusters=K).fit(X)

pred_label = kmeans.predict(X)



N0 = 20;
X1 = np.zeros((N0*K, 784))
X2 = np.zeros((N0*K, 784))

for k in range(K):
	Xk = X[pred_label == k, :]

	center_k = kmeans.cluster_centers_[k]
	neigh = NearestNeighbors(N0)
	neigh.fit(Xk)	
	X1[N0*k: N0*k + N0,:] = Xk[neigh.kneighbors(center_k, N0)[1][0], :]

	X2[N0*k: N0*k + N0,:] = Xk[:N0, :]

# plt.figure(1)

# plt.subplot(121)
A = display_network(kmeans.cluster_centers_.T, 10, 1)

f1 = plt.imshow(A, interpolation='nearest' )
plt.axis('off')

f1.axes.get_xaxis().set_visible(False)
f1.axes.get_yaxis().set_visible(False)
plt.savefig('a.png', bbox_inches='tight')


A = display_network(X2.T, 10, 20)
f2 = plt.imshow(A, interpolation='nearest' )
plt.gray()
# plt.axis('off')
f1.axes.get_xaxis().set_visible(False)
f1.axes.get_yaxis().set_visible(False)
plt.savefig('b.png', bbox_inches='tight')
# plt.show()


# plt.figure(2)
# plt.subplot(122)
A = display_network(X1.T, 10, 20)
f2 = plt.imshow(A, interpolation='nearest' )
plt.gray()
# plt.axis('off')
f1.axes.get_xaxis().set_visible(False)
f1.axes.get_yaxis().set_visible(False)
plt.savefig('c.png', bbox_inches='tight')
