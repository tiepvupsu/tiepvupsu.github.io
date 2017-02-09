
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import linear_model 
from sklearn.metrics import accuracy_score
from scipy import misc      # for 
from sklearn import preprocessing
np.random.seed(1)
path = '../data/AR/' # path to the database 

train_ids = np.arange(1, 26)
test_ids = np.arange(26, 50)
view_ids = np.hstack((np.arange(1, 8), np.arange(14, 21)))

D = 165*120 # original dimension 
d = 500 # new dimension 

# generate the projection matrix 
ProjectionMatrix = np.random.randn(D, d) 


def build_list_fn(pre, img_ids, view_ids):
    """
    pre = 'M-' or 'W-'
    img_ids: indexes of images
    view_ids: indexes of views
    """
    list_fn = []
    for im_id in img_ids:
        for v_id in view_ids:
            fn = path + pre + str(im_id).zfill(3) + '-' +                 str(v_id).zfill(2) + '.bmp'
            list_fn.append(fn)
    return list_fn 


def rgb2gray(rgb):
#     Y' = 0.299 R + 0.587 G + 0.114 B 
    return rgb[:,:,0]*.299 + rgb[:, :, 1]*.587 + rgb[:, :, 2]*.114

# feature extraction 
def vectorize_img(filename):    
    # load image 
    rgb = misc.imread(filename)
    # convert to gray scale 
    gray = rgb2gray(rgb)
    # vectorization each row is a data point 
    im_vec = gray.reshape(1, D)
    return im_vec 

def build_data_matrix(img_ids, view_ids):
    total_imgs = img_ids.shape[0]*view_ids.shape[0]*2 
        
    X_full = np.zeros((total_imgs, D))
    y = np.hstack((np.zeros((total_imgs/2, )), np.ones((total_imgs/2, ))))
    
    list_fn_m = build_list_fn('M-', img_ids, view_ids)
    list_fn_w = build_list_fn('W-', img_ids, view_ids)
    list_fn = list_fn_m + list_fn_w 
    
    for i in range(len(list_fn)):
        X_full[i, :] = vectorize_img(list_fn[i])

    X = np.dot(X_full, ProjectionMatrix)
    # scale to mean 0 and standard deviation 1 
    X_scaled = preprocessing.scale(X)
    return (X_scaled, y)
                
(X_train, y_train) = build_data_matrix(train_ids, view_ids)
(X_test, y_test) = build_data_matrix(test_ids, view_ids)


logreg = linear_model.LogisticRegression(C=1e5) # just a big number 
logreg.fit(X_train, y_train)


y_pred = logreg.predict(X_test)
print "Accuracy: %.2f %%" %(100*accuracy_score(y_test, y_pred))
