# %reset
import numpy as np 
from mnist import MNIST
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

from display_network import *

mndata = MNIST('../MNIST/')
mndata.load_training()
X = np.asarray(mndata.train_images)
y = np.array(mndata.train_labels.tolist())


def extract_data(X, y, classes):
    """
    X: numpy array, matrix of size (N, d), d is data dim
    y: numpy array, size (N, )
    cls: two lists of labels. For example: 
        cls = [[1, 4, 7], [5, 6, 8]]
    return:
        X: extracted data
        y: extracted label 
            (0 and 1, corresponding to two lists in cls)
    """
    y_res_id = np.array([])
    for i in cls[0]:
        y_res_id = np.hstack((y_res_id, np.where(y == i)[0]))
    n0 = len(y_res_id)

    for i in cls[1]:
        y_res_id = np.hstack((y_res_id, np.where(y == i)[0]))
    n1 = len(y_res_id) - n0 
    y_res_id = y_res_id.astype(int)
    X_res = X[y_res_id, :]/255.0

    y_res = np.asarray([0]*n0 + [1]*n1)
    return (X_res, y_res)

cls = [[7], [1]]
(X_train, y_train) = extract_data(X, y, cls)


print(X_train.shape, type(X_train))

from sklearn import linear_model

logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X_train, y_train)

mndata.load_testing()
Xtest_all = np.asarray(mndata.test_images)
ytest_all = np.array(mndata.test_labels.tolist())
(X_test, y_test) = extract_data(Xtest_all, ytest_all, cls)

y_pred = logreg.predict(X_test)

from sklearn.metrics import accuracy_score
print "Accuracy: %.2f %%" %(100*accuracy_score(y_test, y_pred.tolist()))

print(y_pred[:30])
print(y_test[:30])
print(type(y_pred))
print(type(y_test))
print(np.where((y_pred - y_test) != 0))
mis = np.where((y_pred - y_test) != 0)[0]
Xmis = X_test[mis, :]
# print(Xmis.shape)

plt.axis('off')
A = display_network(Xmis.T)
f2 = plt.imshow(A, interpolation='nearest' )
plt.gray()
plt.show()