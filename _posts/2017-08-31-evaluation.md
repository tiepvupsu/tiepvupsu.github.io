---
layout: post
comments: true
title:  "Bài 33: Các phương pháp đánh giá một hệ thống phân lớp (Phần 1/2)"
title2:  "33. Đánh giá hệ thống phân lớp (1/2)"
date:   2017-08-31 15:22:00
permalink: 2017/08/31/evaluation/
mathjax: true
tags: Classification 
category: Classification
sc_project: 11435469
sc_security: 4f6cab76
img: /assets/33_evaluation/roc.png
summary: Các phương pháp đánh giá hiệu năng của một mô hình phân lớp. 
---
**Trong trang này:**

<!-- MarkdownTOC -->

- [1. Giới thiệu](#-gioi-thieu)
- [2. Accuracy](#-accuracy)
- [3. Confusion matrix](#-confusion-matrix)
- [4. True/False Positive/Negative](#-truefalse-positivenegative)
    - [4.1. True/False Positive/Negative](#-truefalse-positivenegative-1)
    - [4.2. Receiver Operating Characteristic curve](#-receiver-operating-characteristic-curve)
    - [4.3. Area Under the Curve](#-area-under-the-curve)
- [5. Tài liệu tham khảo](#-tai-lieu-tham-khao)

<!-- /MarkdownTOC -->

_Một bài viết nhanh được thực hiện khi tôi đang bế tắc trong nghiên cứu. Bạn yên tâm, chuyện này xảy ra thường xuyên. Mọi chuyện rồi sẽ ổn thôi._

Bạn có thể download toàn bộ source code dưới dạng Jupyter Notebook [tại đây](https://github.com/tiepvupsu/tiepvupsu.github.io/blob/master/assets/33_evaluation/python/Evaluation%20methods.ipynb).
<a name="-gioi-thieu"></a>

## 1. Giới thiệu 
Khi xây dựng một mô hình Machine Learning, chúng ta cần một phép đánh giá để xem mô hình sử dụng có hiệu quả không và để so sánh khả năng của các mô hình. Trong bài viết này, tôi sẽ giới thiệu các phương pháp đánh giá các mô hình classification. 

Hiệu năng của một mô hình thường được đánh giá dựa trên tập dữ liệu kiểm thử (test data). Cụ thể, giả sử đầu ra của mô hình khi đầu vào là tập kiểm thử được mô tả bởi vector `y_pred` - là vector dự đoán đầu ra với mỗi phần tử là class được dự đoán của một điểm dữ liệu trong tập kiểm thử. Ta cần so sánh giữa vector dự đoán `y_pred` này với vector class _thật_ của dữ liệu, được mô tả bởi vector `y_true`. 

Ví dụ với bài toán có 3 lớp dữ liệu được gán nhãn là `0, 1, 2`. Trong bài toán thực tế, các class có thể có nhãn bất kỳ, không nhất thiết là số, và không nhất thiết bắt đầu từ `0`. Chúng ta hãy tạm giả sử các class được đánh số từ `0` đến `C-1` trong trường hợp có `C` lớp dữ liệu.  Có 10 điểm dữ liệu trong tập kiểm thử với các nhãn thực sự được mô tả bởi `y_true = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]`. Giả sử bộ phân lớp chúng ta đang cần đánh giá dự đoán nhãn cho các điểm này là `y_pred = [0, 1, 0, 2, 1, 1, 0, 2, 1, 2]`. 

Có rất nhiều cách đánh giá một mô hình phân lớp. Tuỳ vào những bài toán khác nhau mà chúng ta sử dụng các phương pháp khác nhau. Các phương pháp thường được sử dụng là: accuracy score, confusion matrix, ROC curve, Area Under the Curve, Precision and Recall, F1 score, Top R error, etc. 

Trong Phần 1 này, tôi sẽ trình bày về accuracy score, confusion matrix, ROC curve, và Area Under the Curve. Các phương pháp còn lại sẽ được trình bày trong Phần 2. 

<a name="-accuracy"></a>

## 2. Accuracy 
Cách đơn giản và hay được sử dụng nhất là _accuracy_ (độ chính xác). Cách đánh giá này đơn giản tính tỉ lệ giữa số điểm được dự đoán đúng và tổng số điểm trong tập dữ liệu kiểm thử. 

Trong ví dụ này, ta có thể đếm được có 5 điểm dữ liệu được dự đoán đúng trên tổng số 10 điểm. Vậy ta kết luận độ chính xác của mô hình là 0.6 (hay 60%). Để ý rằng đây là bài toán với chỉ 3 class, nên độ chính xác nhỏ nhất đã là khoảng 1/3, khi tất cả các điểm được dự đoán là thuộc vào một class nào đó. 


```python
from __future__ import print_function
import numpy as np 

def acc(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    return float(correct)/y_true.shape[0]

y_true = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
y_pred = np.array([0, 1, 0, 2, 1, 1, 0, 2, 1, 2])
print('accuracy = ', acc(y_true, y_pred))
```

```
    accuracy =  0.6
```

Và đây là [cách tính bằng thư viên](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html):


```python
from sklearn.metrics import accuracy_score
print('accuracy = ',accuracy_score(y_true, y_pred))
```

```
    accuracy =  0.6
```

<a name="-confusion-matrix"></a>

## 3. Confusion matrix 
Cách tính sử dụng accuracy như ở trên chỉ cho chúng ta biết được bao nhiêu phần trăm lượng dữ liệu được phân loại đúng mà không chỉ ra được cụ thể mỗi loại được phân loại như thế nào, lớp nào được phân loại đúng nhiều nhất, và dữ liệu thuộc lớp nào thường bị phân loại nhầm vào lớp khác. Để có thể đánh giá được các giá trị này, chúng ta sử dụng một ma trận được gọi là _confusion matrix_. 

Về cơ bản, confusion matrix thể hiện có bao nhiêu điểm dữ liệu _thực sự_ thuộc vào một class, và được _dự đoán_ là rơi vào một class. Để hiểu rõ hơn, hãy xem bảng dưới đây:

```
 Total: 10 | Predicted | Predicted | Predicted |   
           |    as: 0  |    as: 1  |    as: 2  |   
-----------|-----------|-----------|-----------|---
 True: 0   |     2     |     1     |     1     | 4 
-----------|-----------|-----------|-----------|---
 True: 1   |     1     |     2     |     0     | 3 
-----------|-----------|-----------|-----------|---
 True: 2   |     0     |     1     |     2     | 3 
-----------|-----------|-----------|-----------|---

```

Có tổng cộng 10 điểm dữ liệu. Chúng ta xét ma trận tạo bởi các giá trị tại vùng 3x3 trung tâm của bảng. 

Ma trận thu được được gọi là _confusion matrix_. Nó là một ma trận vuông với số chiều bằng số lượng lớp dữ liệu. Giá trị tại hàng thứ `i`, cột thứ `j` là số lượng điểm **lẽ ra thuộc vào class `i` nhưng lại được dự đoán là thuộc vào class `j`**. Như vậy, nhìn vào hàng thứ nhất (`0`), ta có thể thấy được rằng trong số bốn điểm thực sự thuộc lớp `0`, chỉ có hai điểm được phân loại đúng, hai điểm còn lại bị phân loại nhầm vào lớp `1` và lớp `2`. 

_**Chú ý:** Có một số tài liệu định nghĩa ngược lại, tức giá trị tại **cột** thứ `i`, **hàng** thứ `j` là số lượng điểm lẽ ra thuộc vào class `i` nhưng lại được dự đoán là thuộc vào class `j`. Khi đó ta sẽ được confusion matrix là ma trận chuyển vị của confusion matrix như cách tôi đang làm. Tôi chọn cách này vì đây chính là cách thư viện sklearn sử dụng._

Chúng ta có thể suy ra ngay rằng tổng các phần tử trong toàn ma trận này chính là số điểm trong tập kiểm thử. Các phần tử trên đường chéo  của ma trận là số điểm được phân loại đúng của mỗi lớp dữ liệu. Từ đây có thể suy ra _accuracy_ chính bằng tổng các phần tử trên đường chéo chia cho tổng các phần tử của toàn ma trận. Đoạn code dưới đây mô tả cách tính confusion matrix:


```python
def my_confusion_matrix(y_true, y_pred):
    N = np.unique(y_true).shape[0] # number of classes 
    cm = np.zeros((N, N))
    for n in range(y_true.shape[0]):
        cm[y_true[n], y_pred[n]] += 1
    return cm 

cnf_matrix = my_confusion_matrix(y_true, y_pred)
print('Confusion matrix:')
print(cnf_matrix)
print('\nAccuracy:', np.diagonal(cnf_matrix).sum()/cnf_matrix.sum())

```

    Confusion matrix:
    [[ 2.  1.  1.]
     [ 1.  2.  0.]
     [ 0.  1.  2.]]
    
    Accuracy: 0.6


Cách biểu diễn trên đây của confusion matrix còn được gọi là _unnormalized confusion matrix_, tức ma _confusion matrix_ chưa chuẩn hoá. Để có cái nhìn rõ hơn, ta có thể dùng _normalized confuion matrix_, tức _confusion matrix_ được chuẩn hoá. Để có _normalized confusion matrix_, ta lấy mỗi hàng của _unnormalized confusion matrix_ sẽ được chia cho tổng các phần tử trên hàng đó. Như vậy, ta có nhận xét rằng tổng các phần tử trên một hàng của _normalized confusion matrix_ luôn bằng 1. Điều này thường không đúng trên mỗi cột. Dưới đây là cách tính _normalized confusion matrix_:


```python
normalized_confusion_matrix = cnf_matrix/cnf_matrix.sum(axis = 1, keepdims = True)
print('\nConfusion matrix (with normalizatrion:)')
print(normalized_confusion_matrix)
```

```
    Confusion matrix (with normalizatrion:)
    [[ 0.5         0.25        0.25      ]
     [ 0.33333333  0.66666667  0.        ]
     [ 0.          0.33333333  0.66666667]]
```

Và cách tính [sử dụng thư viện](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html):


```python
from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(y_true, y_pred)
print('Confusion matrix:')
print(cnf_matrix)
```

    Confusion matrix:
    [[2 1 1]
     [1 2 0]
     [0 1 2]]


Confusion matrix thường được minh hoạ bằng màu sắc để có cái nhìn rõ ràng hơn. Đoạn code dưới đây giúp hiển thị confusion matrix ở cả hai dạng (Nguồn: [Confusion matrix](http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html)): 


```python
import matplotlib.pyplot as plt
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims = True)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Plot non-normalized confusion matrix
class_names = [0, 1, 2]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
```

<hr>
<div>
<table width = "100%" style = "border: 0px solid white">
   <tr >
        <td width="40%" style = "border: 0px solid white">
        <img style="display:block;" width = "100%" src = "/assets/33_evaluation/ucm.png">
         </td>
        <td width="40%" style = "border: 0px solid white">
        <img style="display:block;" width = "100%" src = "/assets/33_evaluation/ncm.png">
        </td>

    </tr>

</table>
<div class = "thecap"> Hình 1: Minh hoạ <em> unnormalized confusion matrix </em> và <em> normalized confusion matrix </em>.
</div>
</div>
<hr>


Với các bài toán với nhiều lớp dữ liệu, cách biểu diễn bằng màu này rất hữu ích. Các ô màu đậm thể hiện các giá trị cao. Một mô hình tốt sẽ cho một confusion matrix có các phần tử trên đường chéo chính có giá trị lớn, các phần tử còn lại có giá trị nhỏ. Nói cách khác, khi biểu diễn bằng màu sắc, đường chéo có màu càng đậm so với phần còn lại sẽ càng tốt. Từ hai hình trên ta thấy rằng confusion matrix đã chuẩn hoá mang nhiều thông tin hơn. Sự khác nhau được thấy ở ô trên cùng bên trái. Lớp dữ liệu `0` được phân loại không thực sự tốt nhưng trong _unnormalized confusion matrix_, nó vẫn có màu đậm như hai ô còn lại trên đường chéo chính. 


<a name="-truefalse-positivenegative"></a>

## 4. True/False Positive/Negative 
<a name="-truefalse-positivenegative-1"></a>

### 4.1. True/False Positive/Negative 
Cách đánh giá này thường được áp dụng cho các bài toán phân lớp có hai lớp dữ liệu. Cụ thể hơn, trong hai lớp dữ liệu này có một lớp _nghiêm trọng_ hơn lớp kia và cần được dự đoán chính xác. Ví dụ, trong bài toán xác định có bệnh ung thư hay không thì việc không bị _sót_ (miss) quan trọng hơn là việc chẩn đoán nhầm _âm tính_ thành _dương tính_. Trong bài toán xác định có mìn dưới lòng đất hay không thì việc _bỏ sót_ nghiêm trọng hơn việc _báo động nhầm_ rất nhiều. Hay trong bài toán lọc email rác thì việc cho nhầm email quan trọng vào thùng rác nghiêm trọng hơn việc xác định một email rác là email thường. 

Trong những bài toán này, người ta thường định nghĩa lớp dữ liệu _quan trọng_ hơn cần được xác định đúng là lớp _Positive_ (P-dương tính), lớp còn lại được gọi là _Negative_ (N-âm tính). Ta định nghĩa _True Positive (TP), False Positive (FP), True Negative (TN), False Negative (FN)_ dựa trên _confusion matrix_ chưa chuẩn hoá như sau: 
```
                  |      Predicted      |      Predicted      |
                  |     as Positive     |     as Negative     |
------------------|---------------------|---------------------|
 Actual: Positive | True Positive (TP)  | False Negative (FN) |
------------------|---------------------|---------------------|
 Actual: Negative | False Positive (FP) | True Negative (TN)  |
------------------|---------------------|---------------------|
```

Người ta thường quan tâm đến TPR, FNR, FPR, TNR (R - Rate) dựa trên _normalized confusion matrix_ như sau: 

```
                  |     Predicted      |     Predicted      |
                  |    as Positive     |    as Negative     |
------------------|--------------------|--------------------|
 Actual: Positive | TPR = TP/(TP + FN) | FNR = FN/(TP + FN) |
------------------|--------------------|--------------------|
 Actual: Negative | FPR = FP/(FP + TN) | TNR = TN/(FP + TN) |
------------------|--------------------|--------------------|
```

_False Positive Rate_ còn được gọi là _False Alarm Rate_ (tỉ lệ báo động nhầm), _False Negative Rate_ còn được gọi là _Miss Detection Rate_ (tỉ lệ bỏ sót). Trong bài toán dò mìn, _thà báo nhầm còn hơn bỏ sót_, tức là ta có thể chấp nhận _False Alarm Rate_ cao để đạt được _Miss Detection Rate_ thấp. 

**Chú ý:**:
* Việc biết một cột của confusion matrix này sẽ suy ra được cột còn lại vì tổng các hàng luôn bằng 1 và chỉ có hai lớp dữ liệu.

* **Với các bài toán có nhiều lớp dữ liệu**, ta có thể xây dựng bảng True/False Positive/Negative cho **mỗi lớp** nếu coi lớp đó là lớp _Positive_, các lớp còn lại gộp chung thành lớp _Negative_, giống như cách làm trong [one-vs-rest](https://machinelearningcoban.com/2017/02/11/binaryclassifiers/#one-vs-rest-hay-one-hot-coding). Bạn có thể xem thêm ví dụ [tại đây](http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py). 

<a name="-receiver-operating-characteristic-curve"></a>

### 4.2. Receiver Operating Characteristic curve 
Trong một số bài toán, việc tăng hay giảm FNR, FPR có thể được thực hiện bằng việc thay đổi một _ngưỡng_ (threshold) nào đó. Lấy ví dụ khi ta sử dụng thuật toán [Logistic Regression](https://machinelearningcoban.com/2017/01/27/logisticregression/), đầu ra của mô hình có thể là các _lớp cứng_ `0` hay `1`, hoặc cũng có thể là các giá trị thể hiện xác suất để dữ liệu đầu vào thuộc vào lớp `1`. Khi sử dụng thư viện [sklearn Logistic Regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), ta có thể lấy được các giá trị xác xuất này bằng phương thức `predict_proba()`. Mặc định, ngưỡng được sử dụng là 0.5, tức là một điểm dữ liệu `x` sẽ được dự đoán rơi vào lớp `1` nếu giá trị `predict_proba(x)` lớn hơn 0.5 và ngược lại. 

Nếu bây giờ ta coi lớp `1` là lớp _Positive_, lớp `0` là lớp _Negative_, câu hỏi đặt ra là làm thế nào để tăng mức độ _báo nhầm_ (FPR) để giảm mức độ _bỏ sót_ (FNR)? Chú ý rằng tăng FNR đồng nghĩa với việc giảm TPR vì tổng của chúng luôn bằng 1.

Một kỹ thuật đơn giản là ta thay giá trị threshold từ 0.5 xuống một số nhỏ hơn. Chẳng hạn nếu chọn threshold = 0.3, thì mọi điểm được dự đoán có xác suất đầu ra lớn hơn 0.3 sẽ được dự đoán là thuộc lớp Positive. Nói cách khác, tỉ lệ các điểm được phân loại là Positive sẽ tăng lên, kéo theo cả False Positive Rate và True Positive Rate cùng tăng lên (cột thứ nhất trong ma trận tăng lên). Từ đây suy ra cả FNR và TNR đều giảm. 

Ngược lại, nếu ta muốn _bỏ sót còn hơn báo nhầm_, tất nhiên là ở mức độ nào đó, như bài toán xác định email rác chẳng hạn, ta cần tăng threshold lên một số lớn hơn 0.5. Khi đó, hầu hết các điểm dữ liệu sẽ được dự đoán thuộc lớp `0`, tức _Negative_, và cả TNF và FNR đều tăng lên, tức TPR và FPR giảm xuống. 

Như vậy, ứng với mỗi giá trị của threshold, ta sẽ thu được một cặp (FPR, TPR). Biểu diễn các điểm (FPR, TPR) trên đồ thị khi thay đổi threshold từ 0 tới 1 ta sẽ thu được một đường được gọi là _Receiver Operating Characteristic curve_ hay ROC curve. (_Chú ý rằng khoảng giá trị của threshold không nhất thiết từ 0 tới 1 trong các bài toán tổng quát. Khoảng giá trị này cần được đảm bảo có trường hợp TPR/FPR nhận giá trị lớn nhất hay nhỏ nhất mà nó có thể đạt được_).

Dưới đây là một ví dụ với hai lớp dữ liệu. Lớp thứ nhất là lớp _Negative_ có 20 điểm dữ liệu, 30 điểm còn lại thuộc lớp _Positive_. Giả sử mô hình đang xét cho các đầu ra của dữ liệu được lưu ở biến `scores`. 



```python
# generate simulated data
n0, n1 = 20, 30
score0 = np.random.rand(n0)/2
label0 = np.zeros(n0, dtype = int)
score1  = np.random.rand(n1)/2 + .2
label1 = np.ones(n1, dtype = int)
scores = np.concatenate((score0, score1))
y_true = np.concatenate((label0, label1))

print('True labels:')
print(y_true)
print('\nScores:')
print(scores)
```

```
    True labels:
    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     1 1 1 1 1 1 1 1 1 1 1 1 1]
    
    Scores:
    [ 0.16987517  0.27608323  0.10851568  0.13395249  0.24878687  0.29100097
      0.21036182  0.48215779  0.01930099  0.30927599  0.26581374  0.15141354
      0.26298063  0.10405583  0.30773121  0.39830016  0.04868077  0.17290186
      0.28717646  0.3340749   0.4174846   0.27292017  0.68740357  0.62108568
      0.20781968  0.43056031  0.67816027  0.47037842  0.23118192  0.68862749
      0.24559788  0.58645887  0.69637251  0.5247967   0.24265087  0.60485646
      0.54800088  0.69565411  0.20509934  0.39638029  0.30860676  0.6267616
      0.42360257  0.5507021   0.50313701  0.67614457  0.60108083  0.25201502
      0.27830655  0.58669514]
```

Nhìn chung, các điểm thuộc lớp `1` có `score` cao hơn. Thư viện sklearn sẽ giúp chúng ta tính các thresholds cũng như FPR và TPR tương ứng:


```python
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label = 1)
print('Thresholds:')
print(thresholds)
```

```
    Thresholds:
    [ 0.69637251  0.50313701  0.48215779  0.4174846   0.39830016  0.39638029
      0.30927599  0.30860676  0.28717646  0.27830655  0.27608323  0.27292017
      0.26298063  0.25201502  0.24878687  0.23118192  0.21036182  0.20509934
      0.01930099]
```


```python
print('False Positive Rate:')
print(fpr)
```

```
    False Positive Rate:
    [ 0.    0.    0.05  0.05  0.1   0.1   0.2   0.2   0.35  0.35  0.4   0.4
      0.5   0.5   0.55  0.55  0.6   0.6   1.  ]
```


```python
print('True Positive Rate:')
tpr
```

```
    True Positive Rate:
    array([ 0.03333333,  0.53333333,  0.53333333,  0.66666667,  0.66666667,
            0.7       ,  0.7       ,  0.73333333,  0.73333333,  0.76666667,
            0.76666667,  0.8       ,  0.8       ,  0.83333333,  0.83333333,
            0.93333333,  0.93333333,  1.        ,  1.        ])
```


Như vậy, ứng với `threshold = 0.69637251`, `fpr = 0` và `tpr = 0.03`. Đây không phải là một ngưỡng tốt vì mặc dụ False Positive Rate thấp, True Positive Rate cũng rất thấp. Chúng ta luôn muốn rằng FPR thấp và TPR cao. 

ROC cho bài toán này được minh hoạ như dưới đây:


```python
import matplotlib.pyplot as plt
from itertools import cycle
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
```

<hr>
<table width = "100%" style = "border: 0px solid white">
   <tr >
        <td width="40%" style = "border: 0px solid white">
        <img style="display:block;" width = "100%" src = "/assets/33_evaluation/roc.png">
         </td>
        <td width="40%" style = "border: 0px solid white">
        Hình 2: Ví dụ về Receiver Operating Characteristic curve và Area Under the Curve. 
        </td>
    </tr>
</table>
<hr>


<a name="-area-under-the-curve"></a>

### 4.3. Area Under the Curve
Dựa trên ROC curve, ta có thể chỉ ra rằng một mô hình có hiệu quả hay không. Một mô hình hiệu quả khi có FPR thấp và TPR cao, tức tồn tại một điểm trên ROC curve gần với điểm có toạ độ (0, 1) trên đồ thị (góc trên bên trái). Curve càng gần thì mô hình càng hiệu quả. 

Có một thông số nữa dùng để đánh giá mà tôi đã sử dụng ở trên được gọi là _Area Under the Curve_ hay _AUC_. Đại lượng này chính là diện tích nằm dưới ROC curve màu cam. Giá trị này là một số dương nhỏ hơn hoặc bằng 1. Giá trị này càng lớn thì mô hình càng tốt. 

**Chú ý:** [Cross validation](https://machinelearningcoban.com/2017/03/04/overfitting/#-cross-validation) cũng có thể được thực hiện bằng cách xác định ROC curve và AUC lên [validation set]. 

<a name="-tai-lieu-tham-khao"></a>

## 5. Tài liệu tham khảo
[1] [Sklearn: Receiver Operating Characteristic (ROC) ](http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html)

[2] [Receiver Operating Characteristic (ROC) with cross validation](http://scikit-learn.org/stable/auto_examples/model_selection/)plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
