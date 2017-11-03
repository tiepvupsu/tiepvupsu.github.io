---
layout: post
comments: true
title:  "FundaML 3: Làm việc với các mảng ngẫu nhiên"
title2:  "FundaML 3: Các mảng ngẫu nhiên"
date:   2017-10-28
permalink: 2017/10/20/fundaml_vectors/
mathjax: true
tags: Numpy
category: FundaML
sc_project: 11485460
sc_security: 33a36442
img: /assets/fundaml/matrix.png
summary: Các số ngẫu nhiên đóng một vài trò cực kỳ quan trọng trong lập trình nói chung và lập trình Machine Learning nói riêng. 
---

**Tất cả các bài tập trong bài viết này có thể được thực hiện trực tiếp trên trình duyện qua trang web [FundaML](https://fundaml.com)**

Các số ngẫu nhiên đóng một vài trò cực kỳ quan trọng trong lập trình nói chung và lập trình Machine Learning nói riêng. 

Trong bài học này, chúng ta cùng làm quen với các hàm tạo các số ngẫu nhiên cơ bản.

## 3.1. Mảng ngẫu nhiên các số tuân theo phân bố đều
Một trong những điều quan trọng nhất khi lập trình một ngôn ngữ bất kỳ là cách
sử dụng các hàm ngẫu nhiên. Trong bài này, chúng ta sẽ làm quen tới các hàm
ngẫu nhiên trong Numpy và các cách sử dụng chúng trong các bài toán Machine
Learning.

### 3.1.1. Hàm [`numpy.random.rand`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.random.rand.html)

Hàm `numpy.random.rand` trả về một mảng các số ngẫu nhiên mà mỗi phần tử là một
số ngẫu nhiên có _phân bố đều_ (_uniform distribution_) trong nửa đoạn `[0, 1)`:

```python
>>> import numpy as np 
>>> np.random.rand() 
0.38919680466308004
>>> np.random.rand(3)
array([ 0.48677611,  0.70819795,  0.32393605])
>>> np.random.rand(3, 2)
array([[ 0.29713565,  0.57377171],
       [ 0.0365262 ,  0.04146013],
       [ 0.63039945,  0.8643891 ]])
```

* Nếu số lượng input là 0, hàm trả về một số vô hướng.

* Nếu có inputs (là các số nguyên dương), hàm này trả về một mảng ngẫu nhiên
có số chiều bằng với số inputs, kích thước mỗi chiều bằng với giá trị của các inputs.

### 3.1.2. Hàm [`np.random.seed`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.random.seed.html) 
Các ngôn ngữ lập trình nói chung không tạo ra các giá trị 'thực sự ngẫu nhiên'.
Thật vậy, nếu bạn mở python và bắt đầu với:
```python
>>> import numpy as np
>>> np.random.rand()
```

thì kết quả luôn là các số giống nhau ở mỗi lần thử (bạn hãy thoát python và thử
lại nhiều lần xem). Như trên máy tính của tôi, kết quả lúc nào cũng là
`0.38919680466308004`. _Như vậy, hàm ngẫu nhiên không thực sự sinh ra các giá trị ngẫu nhiên._ Tuy nhiên, nếu thực hiện hàm này rất nhiều lần, chúng ta sẽ thu được các các số
nằm trong khoảng `[0, 1)` mà xác suất để một điểm nằm trong đoạn `[a, b]` với
`0 <= a < b < 1` bằng `b - a`.

Hàm `np.random.seed()` là một hàm được coi như giúp khởi tạo các bộ sinh số ngẫu
nhiên (random generator). Biến số trong seed thường là một số nguyên không
âm 32 bit. Với các giá trị của biến số khác nhau thì sẽ cho ra các số ngẫu
nhiên khác nhau.

Hàm số này được dùng để đối chiều kết quả trong các lần chạy khác nhau trong
các bài toán Machine Learning. Rất nhiều các thuật toán Machine Learning chạy
dựa trên việc tính toán ngẫu nhiên (ví dụ, [Stochastic Gradient Descent](https://machinelearningcoban.com/2017/01/16/gradientdescent2/#-stochastic-gradient-descent)
được sử dụng rất nhiều trong các thuật toán tối ưu Neural Networks). Để đối
chiếu kết quả trong nhiều lần chạy trên, người ta thường khởi tạo các random
generator với các `seed` như nhau.

Các bạn có thể để ý thấy rằng trong các bài trước tôi thường dùng
`np.random.seed()`. Việc đó để đảm bảo rằng kết quả bạn tìm được giống với kết
quả trong code mẫu.

-----------
**Bài tập:** Cho các số `a, b, m, n` trong đó `a < b` là hai số thực bất kỳ; 
`m`, `n` là các số nguyên dương. Viết hàm số tạo một mảng hai chiều có 
`shape = (m, n)`, các phần tử là các số ngẫu nhiên phân bố đều trong 
nửa đoạn `[a, b)`. 

**Chú ý:** 
1. Để kiểm tra mảng trả về có đúng là mảng ngẫu nhiên các phần tử 
trong nửa đoạn \\([a, b)\\) hay không, tôi sẽ tính kỳ vọng (mean) và phương sai 
(variance) của các phần tử trong mảng đó. Tôi biết rằng nếu \\(X\\) là một biến 
ngẫu nhiên tuân theo phân phối chuẩn trong nửa đoạn \\([a, b)\\) thì nó sẽ có kỳ 
vọng và phương sai lần lượt là: 
\\[
\frac{b+a}{2}; \quad \frac{(b-a)^2}{12} 
\\] 

__Lưu ý rằng đây chỉ là điều kiện cần, không phải điều kiện đủ.__

2. Nếu `X` là một biến ngẫu nhiên tuân theo phân phối chuẩn trong nửa đoạn 
`[0, 1)` thì `Y = aX + b` là một biến ngẫu nhiên tuân theo phân phối chuẩn 
trong nửa đoạn `[b, a + b)` nếu `a` là một số dương, hoặc `[a+b, b)` nếu `a` 
là một số âm. 

----------



## 3.2. Mảng ngẫu nhiên các số tuân theo phân phối chuẩn 
[Phân phối chuẩn](https://en.wikipedia.org/wiki/Normal_distribution) 
(normal distribution) hay phân bố Gassian (Gassian distribution)
rất quan trọng trong thực tế và các bài toán kỹ thuật. 

Hàm `numpy.random.randn()` (chữ `n` ở cuối là viết tắt của _normal_) có chức 
năng tương tự như hàm `np.random.rand` nhưng kết quả trả về là mảng có các phần 
tử phân bố theo phân phối chuẩn có kỳ vọng bằng 0 và phương sai bằng 1.
```python 
>>> np.random.randn()
-0.4718968059633623
>>> np.random.randn(3)
array([ 0.73658734,  1.1116358 , -0.82687362])
>>> np.random.randn(3, 2)
array([[-1.03072303, -2.48099731],
       [-0.23800829,  0.56195321],
       [ 0.74327256, -1.22951965]])
```

------------------

**Bài tập:** 
Cho các số `a, s, m, n` với: 
* `a` là một số thực bất kỳ. 
* `s` là một số thực dương. 
* `m, n` là các số nguyên dương. 

Xây dựng một mảng ngẫu nhiên hai chiều có `shape = (m, n)` mà các phần tử của nó 
tuần theo phân phối chuẩn có kỳ vọng bằng `a` và phương sai là `s`. 

**Chú ý:** Ký hiệu \\(\mathcal{N}(\mu, \sigma^2)\\) để chỉ một phân phối chuẩn có 
kỳ vọng \\(\mu\\) và phương sai \\(\sigma^2\\). Một biến ngẫu nhiên \\(X\\) tuân theo 
phân phối chuẩn có kỳ vọng \\(\mu\\), phương sai \\(\sigma^2\\) sẽ được ký hiệu là 
\\(X \sim \mathcal{N}(\mu, \sigma^2)\\). 

Nếu \\(X \sim \mathcal{N}(\mu, \sigma^2)\\) thì: 
* \\(X+a \sim \mathcal{N}( \mu+a, \sigma^2)\\) với \\(a\\) là một số thực bất kỳ. 
* \\(kX \sim \mathcal{N}(k\mu, k^2\sigma^2)\\) với \\(k\\) là một số thực bất kỳ. 

------------------



## 3.3. Mảng ngẫu nhiên các số nguyên

Hàm tạo mảng các số tự nhiên ngẫu nhiên. Bạn đọc có thể tham khảo trực tiếp
cách sử dụng trong numpy document:

```
>>> import numpy as np 
>>> help(np.random.randint)
randint(low, high=None, size=None, dtype='l')
    
    Return random integers from `low` (inclusive) to `high` (exclusive).
    
    Return random integers from the "discrete uniform" distribution of
    the specified dtype in the "half-open" interval [`low`, `high`). If
    `high` is None (the default), then results are from [0, `low`).
```

Chú ý cụm `"discrete uniform" distribution`. Điều này tức là mỗi số nguyên
trong nửa đoạn [`low`, `high`) sẽ xuất hiện với xác suất bằng nhau.
Ví dụ: 

```python 
>>> np.random.randint(1, 3)
2
>>> # a 2-by-4 np array with random integer elements in {3, 4}
>>> np.random.randint(3, 5, (2, 4))
array([[3, 3, 4, 4],
       [3, 4, 3, 4]])
```

_Phần này không có bài tập._

## 3.4. Hoán vị 
```python 
>>> import numpy as np 
>>> np.random.permutation(10)
array([2, 4, 3, 8, 5, 0, 1, 7, 6, 9])
```

Ví dụ trên đây có mục đích là tạo ra một mảng có 10 phần tử bao gồm các số tự
nhiên từ 0 đến 9 sắp xếp theo thứ tự ngẫu nhiên. Mảng này còn được gọi là một
hoán vị của các số từ 0 đến 9. 

Hoán vị ngẫu nhiên được sử dụng rất nhiều khi xử lý dữ liệu trong Machine
Learning. Dưới đây là hai ví dụ điển hình. 

### 3.4.1. Stochastic Gradient Descent. 

Trong [Stochastic Gradient Descent](https://machinelearningcoban.com/2017/01/16/gradientdescent2/#-stochastic-gradient-descent),
việc quan trọng nhất là ở mỗi epoch, chúng ta cần *trộn lẫn* thứ tự của dữ liệu 
và lấy ra từng mini-batch trong đó. Cụ thể, nếu coi toàn bộ dữ liệu là một ma 
trận, mỗi hàng là một điểm dữ liệu và có tổng cộng \\(N\\) điểm. Tại mỗi 
iteration, ta sẽ lấy ra một tập con \\(k\\) điểm dữ liệu, với \\(k \ll N\\) để 
cập nhật nghiệm. Trong một epoch, ta cần đảm bảo rằng tất cả các điểm dữ liệu 
đều được lấy ra tại một minibatch nào đó và không có điểm nào được lấy quá một 
lần (giả sử rằng \\(N\\) chia hết cho \\(k\\)). Và hơn nữa, việc lấy ra các 
minibatch ở mỗi epoch là khác nhau.

Việc này có thể được thực hiện bằng cách tạo ra một hoán vị ngẫu nhiên của các 
số từ \\(0\\)( đến \\(N-1\\) và coi chúng như chỉ số của các điểm dữ liệu. Tại 
minibatch thứ nhất, ta lấy ra các hàng có chỉ số tương ứng với \\(k\\) số đầu 
tiên trong hoán vị tìm được. Lần lượt như vậy cho tới khi minibatch cuối cùng 
được lấy ra. Sau đó ta lại *trộn lẫn* dữ liệu bằng một hoán vị ngẫu nhiên khác.

### 3.4.2. Chia dữ liệu training và test 
(Bạn đọc có thể tham khảo cách trực tiếp sử dụng thư viện [tại đây](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html))

Khi kiểm tra một thuật toán Machine Learning, người ta thường chia tập dữ liệu
thu được thành hai phần: training và test (có thể có thêm validation). Một điều
quan trọng là phần phân chia này phải được tạo một cách ngẫu nhiên để tránh việc
dữ liệu được phân chia một cách quá thiên lệch (_biased_). Và đây là lúc chúng
ta có thể sử dụng các hoán vị ngẫu nhiên. 

Giả sử có 100 điểm dữ liệu, ta cần lấy ngẫu nhiên ra 70 điểm làm training test,
30 điểm còn lại làm test set. Cách đơn giản nhất là tạo một hoán vị ngẫu nhiên
của các số từ 0 đến 99. Sau đó 70 điểm có chỉ số là 70 phần tử đầu tiên của mảng
hoán vị được dùng làm training set, 30 điểm còn lại được dùng làm test set. 

------------

**Bài tập:**
Cho hai số tự nhiên `N > k > 0` viết hàm số `sample_no_replace(N, k)` trả về
ngẫu nhiên `k` số tự nhiên nằm trong tập `{0, 1, ..., N-1}` sao cho không có hai
số nào trùng nhau. 

Việc *ngẫu nhiên* ở đây sẽ được kiểm chứng bằng cách gọi hàm 
`sample_no_replace(N, k)` nhiều lần. Trong toàn bộ các kết quả trả về, tần suất 
xuất hiện của mỗi số trong tập `{0, 1, ..., N-1}` phải gần bằng nhau.

_Giả sử `X` là ma trận chứa `N` điểm dữ liệu theo hàng. Nếu 
`idx =sample_no_replace(N, k)` là kết quả trả về của hàm bạn đã viết, `k` điểm 
ngẫu nhiên của `X` có thể được lấy ra bằng `X[idx, :]`._

------------

(_còn nữa_)

