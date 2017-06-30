import numpy as np
from sklearn import neighbors, datasets
from mnist import MNIST # require `pip install python-mnist`

from sklearn.model_selection import train_test_split
# https://pypi.python.org/pypi/python-mnist/

# you need to download the MNIST dataset first
# at: http://yann.lecun.com/exdb/mnist/
mndata = MNIST('../MNIST/') # path to your MNIST folder 
mndata.load_testing()
mndata.load_training()
X = mndata.test_images
# X_train = mndata.train_images
y = np.asarray(mndata.test_labels)
# y_train = np.asarray(mndata.train_labels)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=5000)

from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# dimensionality reduction by LDA 
clf = LinearDiscriminantAnalysis()
clf.fit(X_train, y_train)

# LDA classifier
y_pred = clf.predict(X_test)
print("LDA classifier accuracy: %.2f %%" %(100*accuracy_score(y_test, y_pred)))

# Train a kernel SVM on projected data by LDA
X_train_new = clf.transform(X_train)
X_test_new = clf.transform(X_test)

print(X_train_new.shape)


from sklearn import svm
svm1 = svm.SVC(kernel = 'linear')
svm1.fit(X_train_new, y_train)

y_pred1= svm1.predict(X_test_new)

print("LDA + SVM accuracy     : %.2f %%" %(100*accuracy_score(y_test, y_pred1)))

# Train a kernel SVM on projected data by PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=20) # K = 100
pca.fit(X_train)

X_train_new = pca.transform(X_train)
X_test_new = pca.transform(X_test)

svm1 = svm.SVC(kernel = 'linear')
svm1.fit(X_train_new, y_train)

y_pred2 = svm1.predict(X_test_new)

print("PCA + SVM accuracy    : %.2f %%" %(100*accuracy_score(y_test, y_pred2)))