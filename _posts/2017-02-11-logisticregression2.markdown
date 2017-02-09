---
layout: post
comments: true
title:  "Bài 12: Binary Classifiers cho các bài toán Classification"
date:   2017-02-11 15:22:00
permalink: 2017/02/11/binaryclassifiers/
mathjax: true
tags: Neural-nets Supervised-learning Regression Binary-classifier Multi-class
category: Neural-nets
sc_project: 
sc_security: 
img: \assets\LogReg2\ARgender.png
summary: Một vài ứng dụng của Logistic Regression. Logistic Regression cho các bài toán multi-class classification. 
---

Cho tới bây giờ, ngoài _thuật toán lười_ [K-nearest neighbors](/2017/01/08/knn/), tôi đã giới thiệu với bạn đọc hai thuật toán cho các bài toán Classification: [Perceptron Learning Algorithm](/2017/01/21/perceptron/) và [Logistic Regression](/2017/01/27/logisticregression/). Hai thuật toán này được xếp vào loại Binary Classifiers vì chúng được xây dựng dựa trên ý tưởng về các bài toán classification với chỉ hai classes. Trong bài viết này, tôi sẽ cùng các bạn làm một vài ví dụ nhỏ về ứng dụng đơn giản (nhưng thú vị) của các binary classifiers, cho cả các bài toán với hai classes, và cách mở rộng chúng để áp dụng cho các bài toán với nhiều classes. 


Vì Logistic Regression không yêu cầu các classes phải [_linearly separable_](/2017/01/21/perceptron/#bai-toan-perceptron) như PLA, nên tôi sẽ sử dụng Logistic Regression để đại diện cho các binary classifiers. _Chú ý rằng, có rất nhiều các thuật toán cho binary classification nữa mà tôi chưa giới thiệu. Tạm thời, với những gì đã viết,tôi chỉ sử dụng Logistic Regression. Các kỹ thuật trong bài viết này hoàn toàn có thể áp dụng cho các binary classifiers khác._


**Trong trang này:** 
<!-- MarkdownTOC -->

- [Bài toán phân biệt giới tính dựa trên ảnh khuôn mặt](#bai-toan-phan-biet-gioi-tinh-dua-tren-anh-khuon-mat)
    - [Làm việc với Python](#lam-viec-voi-python)
- [Bài toán phân biệt hai chữ số viết tay](#bai-toan-phan-biet-hai-chu-so-viet-tay)
- [Binary Classifiers cho Multi-class Classification problems](#binary-classifiers-cho-multi-class-classification-problems)
- [Thảo luận](#thao-luan)
- [Tài liệu tham khảo](#tai-lieu-tham-khao)

<!-- /MarkdownTOC -->


<a name="bai-toan-phan-biet-gioi-tinh-dua-tren-anh-khuon-mat"></a>

## Bài toán phân biệt giới tính dựa trên ảnh khuôn mặt 
Chúng ta cùng bắt đầu với bài toán phân biệt giới tính dựa trên ảnh khuôn mặt. Về ảnh khuôn mặt, bộ cơ sở dữ liệu [AR Face Database](http://www2.ece.ohio-state.edu/~aleix/ARdatabase.html) được sử dụng rộng rãi. 

Bộ cơ sở dữ liệu này bao gồm hơn 4000 ảnh màu tương ứng với khuôn mặt của 126 người (70 nam, 56 nữ). Với mỗi người, 26 bức ảnh được chụp ở các điều kiện ánh sáng khác nhau, sắc thái biểu cảm khuôn mặt khác nhau, và bị che mắt (bởi kính râm) hoặc miệng (bởi khăn); và được chụp tại hai thời điểm khác nhau cách nhau 2 tuần. 

Để cho đơn giản, tôi sử dụng bộ cơ sử AR Face thu gọn (có thể tìm thấy trong cùng trang web phía trên, mục _Other (relevant) downloads_). Bộ cơ sở dữ liệu thu gọn này bao gồm 2600 bức ảnh từ 50 nam và 50 nữ. Hơn nữa, các khuôn mặt cũng đã được xác định chính xác và được _cropped_ với kích thước 165 x 120 (pixel) bằng phương pháp được mô tả trong bài báo [PCA veus LDA](http://lectures.molgen.mpg.de/networkanalysis13/PCAversusLDA_eigenfaces.pdf). Tôi xin bỏ qua phần xử lý này và trực tiếp sử dụng ảnh đã cropped như một số ví dụ dưới đây:

<div class="imgcap">
<img src ="\assets\LogReg2\ARgender.png" align = "center" width = "800">
<div class = "thecap">Hình 1: Các ví dụ mẫu trong AR Face database thu gọn.</div>
</div> 

**Lưu ý:**

* _Vì lý do bản quyền, tôi không được phép chia sẻ với các bạn bộ dữ liệu này. Các bạn muốn sở hữu có thể liên lạc với tác giả như hướng dẫn ở trong website [AR Face Database](http://www2.ece.ohio-state.edu/~aleix/ARdatabase.html). Một khi các bạn đã có tài khoản để download, tôi mong các bạn tôn trọng tác giả và không chia sẻ trực tiếp với bạn bè._

* _Có một cách đơn giản và nhanh hơn để lấy được các feature vector (sau bước [Feature Engineering](/general/2017/02/06/featureengineering/))  của cơ sở dữ liệu này mà không cần liên lạc với tác giả. Các bạn có thể tìm  [tại đây](https://www.umiacs.umd.edu/~zhuolin/projectlcksvd.html), phần **Downloads**, mục **Random face features for AR database**._

Mỗi bức ảnh trong AR Face thu gọn được đặt tên dưới dạng `G-xxx-yy.bmp` Trong đó: `G` nhận một trong hai giá trị `M` (man) hoặc `W` (woman); `xxx` là id của người, nhận gía trị từ `001` đến `050`; `yy` là điều kiện chụp, nhận giá trị từ `01` đến `26`, trong đó các góc từ `01` đến `07` và từ `14` đến `20` là các góc không bị che bởi kính hoặc khăn. 

Để làm ví dụ cho thuật toán Logistic Regression, tôi lấy ảnh của 25 nam và 25 nữ đầu tiên làm tập training set; 25 nam và 25 nữ còn lại làm test set. Với mỗi người, tôi chỉ lấy các khuôn mặt không bị che bởi kính và khăn.

**Feature Extraction**: vì mỗi bức ảnh có kích thước `3x165x120` là một số khá lớn nên ta sẽ làm thực hiện Feature Extraction bằng hai bước đơn giản sau (_bạn đọc được khuyến khích đọc bài [Giới thiệu về Feature Engineering](/general/2017/02/06/featureengineering/)_): 

* Chuyển ảnh màu về ảnh xám theo công thức `Y' = 0.299 R + 0.587 G + 0.114 B ` (Xem thêm tại [Grayscale - wiki](https://en.wikipedia.org/wiki/Grayscale#Luma_coding_in_video_systems)). 

* _Kéo dài_ ảnh xám thu được thành 1 vector hàng có số chiều `165x120`, sau đó sử dụng một _random projection matrix_ để giảm số chiều về `500`. Bạn đọc có thể thay giá trị này bằng các số khác. 

Chúng ta có thể bắt tay làm việc với Python ngay bây giờ. Tôi sẽ sử dụng hàm [sklearn.linear_model.LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) trong thư viện `sklearn` cho các ví dụ trong bài này. Nếu không muốn đọc phần này, bạn có thể lấy [source code ở dây](/assets/LogRegs/ARgender.py). 

**Chú ý:** Hàm [sklearn.linear_model.LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) nhận dữ liệu ở dạng vector hàng. 

<a name="lam-viec-voi-python"></a>

### Làm việc với Python

Khai báo thư viện

```python
import numpy as np 
from sklearn import linear_model           # for logistic regression
from sklearn.metrics import accuracy_score # for evaluation
from scipy import misc                     # for loading image
from sklearn import preprocessing          # for some scaling
np.random.seed(1)                          # for fixing random values
```


Phân chia training set và test set, lựa chọn các _views_.

```python
path = '../data/AR/' # path to the database 
train_ids = np.arange(1, 26)
test_ids = np.arange(26, 50)
view_ids = np.hstack((np.arange(1, 8), np.arange(14, 21)))
```

Tạo _random projection matrix_. 


```python
D = 165*120 # original dimension 
d = 500 # new dimension 

# generate the projection matrix 
ProjectionMatrix = np.random.randn(D, d) 
```

Xây dựng danh sách các tên files.

```python
def build_list_fn(pre, img_ids, view_ids):
    """
    INPUT:
        pre = 'M-' or 'W-'
        img_ids: indexes of images
        view_ids: indexes of views
    OUTPUT:
        a list of filenames 
    """
    list_fn = []
    for im_id in img_ids:
        for v_id in view_ids:
            fn = path + pre + str(im_id).zfill(3) + '-' + \
                str(v_id).zfill(2) + '.bmp'
            list_fn.append(fn)
    return list_fn 
```

**Feature Extraction:** Xây dựng dữ liệu cho training set và test set.

```python
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
    list_fn   = list_fn_m + list_fn_w 
    
    for i in range(len(list_fn)):
        X_full[i, :] = vectorize_img(list_fn[i])

    X = np.dot(X_full, ProjectionMatrix)
    # scale to mean 0 and standard deviation 1 
    X_scaled = preprocessing.scale(X)
    return (X_scaled, y)
                
(X_train, y_train) = build_data_matrix(train_ids, view_ids)
(X_test, y_test)   = build_data_matrix(test_ids, view_ids)
```

Thực hiện thuật toán Logistic Regression, dự đoán output của test data và đánh giá kết quả. Một chú ý nhỏ, hàm Logistic Regression trong thư viện sklearn có nhiều biến thể khác nhau. Để sử dụng thuật toán Logistic Regression _thuần_ mà tôi đã giới thiệu trong Bài 10, chúng ta cần đặt giá trị cho `C` là một số lớn (để nghịch đảo của nó gần với 0. Tạm thời các bạn chưa cần quan tâm tới điều này, chỉ cần chọn `C` lớn là được).


```python
logreg = linear_model.LogisticRegression(C=1e5) # just a big number 
logreg.fit(X_train, y_train)


y_pred = logreg.predict(X_test)
print "Accuracy: %.2f %%" %(100*accuracy_score(y_test, y_pred))
```

    Accuracy: 91.37 %

91.37%, tức là cứ 10 bức ảnh trong test set thì có trung bình hơn 9 bức được nhận dạng đúng. Không tệ, nhất là khi chúng ta vẫn chưa phải làm gì nhiều!

<a name="bai-toan-phan-biet-hai-chu-so-viet-tay"></a>

## Bài toán phân biệt hai chữ số viết tay 

<a name="binary-classifiers-cho-multi-class-classification-problems"></a>

## Binary Classifiers cho Multi-class Classification problems 

<a name="thao-luan"></a>

## Thảo luận 

<a name="tai-lieu-tham-khao"></a>

## Tài liệu tham khảo



