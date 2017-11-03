---
layout: post
comments: true
title:  "FundaML 1: Làm việc với mảng một chiều"
title2:  "FundaML 1: Mảng một chiều"
date:   2017-10-28
permalink: 2017/10/12/fundaml_vectors/
mathjax: true
tags: Numpy
category: FundaML
sc_project: 11485457 
sc_security: 15e00109
img: /assets/fundaml/matrix.png
summary: 
---

**Tất cả các bài tập trong bài viết này có thể được thực hiện trực tiếp trên trình duyện qua trang web [FundaML](https://fundaml.com)**
<!-- MarkdownTOC -->

- [0. Giới thiệu về Numpy](#-gioi-thieu-ve-numpy)
    - [0.1. Cài đặt Numpy](#-cai-dat-numpy)
- [1.1 Khởi tạo mảng 1 chiều](#-khoi-tao-mang--chieu)
    - [1.1.1. Khai báo vector](#-khai-bao-vector)
- [1.2. Kiểu dữ liệu của mảng](#-kieu-du-lieu-cua-mang)
    - [1.2.1. Kiểu dữ liệu](#-kieu-du-lieu)
- [1.3. Khởi tạo các mảng một chiều đặc biệt](#-khoi-tao-cac-mang-mot-chieu-dac-biet)
    - [1.3.1. Mảng toàn giá trị 0 hoặc 1](#-mang-toan-gia-tri--hoac-)
    - [1.3.2. Cấp số cộng](#-cap-so-cong)
- [1.4. Truy cập mảng một chiều](#-truy-cap-mang-mot-chieu)
    - [1.4.1. Kích thước của mảng](#-kich-thuoc-cua-mang)
    - [1.4.2. Chỉ số](#-chi-so)
    - [1.4.3. Đọc từng phần tử của vector](#-doc-tung-phan-tu-cua-vector)
    - [1.4.4. Chỉ số ngược](#-chi-so-nguoc)
    - [1.4.5. Thay đổi giá trị một phần tử của mảng](#-thay-doi-gia-tri-mot-phan-tu-cua-mang)
- [1.5. Truy cập nhiều phần tử của mảng một chiều](#-truy-cap-nhieu-phan-tu-cua-mang-mot-chieu)
    - [1.5.1. Đọc](#-doc)
    - [1.5.2. Ghi](#-ghi)
    - [1.5.3. Đọc thêm](#-doc-them)
- [1.6. Tính toán giữa các mảng một chiều và số vô hướng](#-tinh-toan-giua-cac-mang-mot-chieu-va-so-vo-huong)
    - [1.6.1. Phép toán giữa mảng một chiều với một số vô hướng.](#-phep-toan-giua-mang-mot-chieu-voi-mot-so-vo-huong)
    - [1.6.2. Phép toán giữa hai mảng một chiều](#-phep-toan-giua-hai-mang-mot-chieu)
    - [1.6.3. Các hàm toán học](#-cac-ham-toan-hoc)
- [1.7. Norm 1](#-norm-)
- [1.8. Hàm Softmax cho mảng một chiều](#-ham-softmax-cho-mang-mot-chieu)
- [1.9. Tích vô hướng của hai vectors - Norm 2](#-tich-vo-huong-cua-hai-vectors---norm-)
- [1.10. min, max, armin, argmax của mảng một chiều](#-min-max-armin-argmax-cua-mang-mot-chieu)
    - [1.10.1. min, max](#-min-max)
    - [1.10.2. argmin, argmax](#-argmin-argmax)
- [Softmax II - Phiên bản ổn định](#softmax-ii---phien-ban-on-dinh)

<!-- /MarkdownTOC -->


<a name="-gioi-thieu-ve-numpy"></a>

## 0. Giới thiệu về Numpy 
Mặc dù các bài học trong khoá này có thể được thực hiện trực tiếp trên trình
duyệt, tôi vẫn khuyến khích các bạn cài đặt Python và Numpy vào trong máy
tính cá nhân để việc lập trình được thuận tiện hơn.

Tôi giả sử các bạn đã từng sử dụng Python và có kiến thức cơ bản về Python.
Nếu bạn chưa học Python bao giờ, dưới đây là một vài khoá học và trang web mà
tôi thấy có chất lượng tốt:

1. [Introduction to Computer Science and Programming Using Python](https://www.edx.org/course/introduction-computer-science-mitx-6-00-1x-10)
2. [learnpython.org](https://www.learnpython.org/)

**Chú ý rằng phiên bản Python được sử dụng ở đây là Python 3.**

<a name="-cai-dat-numpy"></a>

### 0.1. Cài đặt Numpy 
Numpy là một thư viện của Python hỗ trợ cho việc tính toán các mảng nhiều
chiều, có kích thước lớn với các hàm số đã được tối ưu áp dụng lên các mảng
nhiều chiều đó. Numpy đặc biệt hữu ích khi thực hiện các hàm số liên quan tới
Đại Số Tuyến Tính.

_Bạn đọc có thể tham khảo [tài liệu về
numpy.](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwiJpuTO-NTVAhWCUBQKHdmqDJ0QFggoMAA&url=http%3A%2F%2Fwww.numpy.org%2F&usg=AFQjCNEN-XKZnvvnUV0ZkdbbQbR-GHVEzg)_

Để cài đặt Numpy và các thư viện thường dùng trong Machine Learning, bạn có
thể tham khảo các bài hướng dẫn bằng Tiếng Việt dưới đây:

* [Hướng dẫn cài đặt python và các thư viện trên MacOS và Linux](https://machinelearningcoban.com/faqs/#-huong-dan-cai-dat-python-va-cac-thu-vien-tren-macos)
* [Hướng dẫn cài đặt python và các thư viện trên Windows](https://machinelearningcoban.com/faqs/#-huong-dan-cai-dat-python-va-cac-thu-vien-tren-windows)

Sau khi cài đặt xong, trong Python, chúng ta cần khai báo: 

```python
import numpy
```
để có thể bắt đầu sử dụng các hàm số của numpy.

Vì numpy là thư viện được sử dụng thường xuyên nên nó thường được khai báo
gọn lại thành `np`:

```python
import numpy as np
```

`np` có thể thay bằng các từ khác (không phải từ khoá), tuy nhiên, bạn được
khuyến khích đặt là `np` vì các tài liệu hướng dẫn đều ngầm quy ước với nhau
như thế.

**Có một điểm đặc biệt cần lưu ý: biến numpy là các biến _mutable_. Bạn cần phân biệt rõ biến [mutable và immutable trong Python](https://en.wikibooks.org/wiki/Python_Programming/Data_Types#Mutable_vs_Immutable_Objects).** 

Tiếp theo, chúng ta sẽ làm quen với cách sử dụng numpy từ đơn giản tới ít đơn
giản hơn. Các bạn có thể di chuyển giữa các bài học thông qua nút "Lesson
Outline" và hai nút điều hướng ở đầu trang [FundaML](https://fundaml.com).

<a name="-khoi-tao-mang--chieu"></a>

## 1.1 Khởi tạo mảng 1 chiều
<a name="-khai-bao-vector"></a>

### 1.1.1. Khai báo vector
Trong Numpy, vector được hiểu là một mảng 1 chiều. 

Ví dụ: để có một vector `x = [1, 2, 3]`, chúng ta thực hiện như sau:
```python
>>> x = np.array([1, 2, 3])
>>> x
array([1, 2, 3])
```
**Chúng ta ngầm hiểu rằng thư viện numpy đã được khai báo bởi:** `import numpy as np`.

Các dòng _không_ bắt đầu với `>>>` là các dòng hiển thị đầu ra.

Xin nhắc lại, Numpy không quy ước vector hàng hay vector cột mà chỉ coi một vector là một mảng một chiều. Nếu bạn thực sự muốn có một vector cột, bạn cần phải coi nó là một
ma trận có số chiều thứ hai bằng 1, và khi đó phải khai báo với numpy rằng đó là một mảng hai chiều. Chúng ta sẽ quay lại vấn đề này trong bài Ma trận.

Để hiểu thêm về hàm`np.array`, bạn có thể xem thêm [numpy.array](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html), hoặc gõ trực tiếp vào cửa sổ dòng lệnh:
```python
>>> help(np.array)
```

Cú pháp `help(func)` khi được thực hiện trên cửa sổ dòng lệnh (terminal), với `func` là tên hàm số, sẽ hiển thị hướng dẫn sử dụng hàm số đó. 

----------
> **Bài tập**: Khởi tạo một vector `x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`. Chú ý chỉ sửa code giữa các dòng bắt đầu bởi `# TODO:` và `# -- end TODO --`.

----

<a name="-kieu-du-lieu-cua-mang"></a>

## 1.2. Kiểu dữ liệu của mảng 

<a name="-kieu-du-lieu"></a>

### 1.2.1. Kiểu dữ liệu
Nếu khai báo:
```python
>>> import numpy as np
>>> a = np.array([1, 2])
>>> print(type(a[0]))
<class 'numpy.int64'>
```

ta sẽ thấy các thành phần của `a` mặc định là kiểu số nguyên `numpy.int64`. Chú ý rằng `type(var)` trả về kiểu dữ liệu của biến `var`.
Để khai báo `a` là mảng với các thành phần là thực, ta cần viết đưới dạng:

```python
>>> a = np.array([1.0, 2.0]) 
>>> # or np.array([1., 2.]) or np.array([1., 2])
>>> print(type(a))
<class 'numpy.float64'>
```

Ta cũng có thể _ép_ kiểu, ví dụ kiểu dữ liệu thực `numpy.float64`, ta có thể sử dụng từ khoá `dtype` như dưới đây:

```python
>>> a = np.array([1, 2], dtype = np.float64)
>>> print(type(a[0]))
<class 'numpy.float64'>
```

___

Phần này không có bài tập.

----

<a name="-khoi-tao-cac-mang-mot-chieu-dac-biet"></a>

## 1.3. Khởi tạo các mảng một chiều đặc biệt 
<a name="-mang-toan-gia-tri--hoac-"></a>

### 1.3.1. Mảng toàn giá trị 0 hoặc 1

Vector \\(\mathbf{0}\\) là một vector đặc biệt được dùng rất thường xuyên để khởi tạo. Để tạo một vector \\(\mathbf{0}\\) có số phần tử là `d`, ta dùng hàm [`numpy.zeros`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html).

```python
>>>> import numpy as np 
>>> np.zeros(3)
array([ 0.,  0.,  0.])
```

Tương tự như thế, với mảng toàn giá trị 1, ta sẽ dùng hàm [`numpy.ones`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ones.html):
```python
>>> np.ones(5)
array([ 1.,  1.,  1.,  1.,  1.]) 
```

Ngoài ra, numpy còn cung cấp hai hàm đặc biệt `numpy.zeros_like` và `numpy.ones_like` giúp tạo các mảng 0 và mảng 1 có số chiều giống như chiều của biến số.

```python
>>> x = np.array([1, 2, 3])
>>> np.zeros_like(x)
array([0, 0, 0])
>>> np.ones_like(x)
array([1, 1, 1])
```

<a name="-cap-so-cong"></a>

### 1.3.2. Cấp số cộng 
Để tạo mảng các số nguyên từ `0` đến `n-1` (`n` số tổng cộng) ta dùng hàm `np.arange(n)`:
```python
>>> np.arange(3)
array([0, 1, 2]) 
```

Để tạo mảng các số nguyên từ `m` đến `n-1`, ta cũng dùng hàm này ở dạng `np.arange(m, n)`:
```python
>>> np.arange(3, 6)
array([3, 4, 5])
```

Để tạo một cấp số cộng với phần tử đầu là `a`, công sai `d` dương, phần tử cuối là số lớn nhất _nhỏ hơn_ `b`, ta dùng `np.arange(a, b, d)`.
```python
>>> np.arange(0, 1, 0.1)
array([ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9])
```

Nếu `d` là 1 số âm và `b < a` thì phần tử cuối là phần tử nhỏ nhất của cấp số cộng _lớn hơn_ `b`:
```python 
>>> np.arange(5, 1, -0.9)
array([ 5. ,  4.1,  3.2,  2.3,  1.4])
```

---
**Bài tập 1:**
Xây dựng mảng các luỹ thừa của 2 nhỏ hơn 1025, bao gồm cả `1 = 2**0`.
_Gợi ý: Nếu `a` là một mảng và `b` là một số thì `b**a` sẽ trả về một mảng cùng kích thước với `a` mà phần tử có chỉ số `i` bằng `b**a[i]`, với `**` là toán tử luỹ thừa._

**Bài tập 2:**
Xây dựng mảng gồm 10 phần tử, trong đó 9 phần tử đầu bằng 3, phần tử cuối cùng bằng 1.5.

---

<a name="-truy-cap-mang-mot-chieu"></a>

## 1.4. Truy cập mảng một chiều 
<a name="-kich-thuoc-cua-mang"></a>

### 1.4.1. Kích thước của mảng 
Kích thước của một mảng numpy `x` nói chung được xác định bằng [`numpy.array.shape`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html). Ví dụ:

```python
>>> x = np.array([3, 4, 5])
>>> print(x.shape)
(3,)
```

Kết quả trả về là một [tuple](https://www.tutorialspoint.com/python/python_tuples.htm). 
Nếu `x` là một mảng một chiều, kết quả trả về sẽ có dạng `(d,)` trong đó `d`, là phần tử đầu tiên-và duy nhất- của tuple này, 
là số phẩn tử của `x`. Chú ý rằng sau số `3` còn dấu `,` nữa để chắc chắn rằng kết quả là 1 tuple. 

Để lấy giá trị `d` này, ta dùng:

```python
>>> d = x.shape[0]
```

<a name="-chi-so"></a>

### 1.4.2. Chỉ số

Mỗi thành phần trong mảng 1 chiều tương ứng với một chỉ số. Chỉ số trong numpy, cũng giống như chỉ số trong python, bắt đầu bằng 0. Nếu mảng 1 chiều có `d` phần tử thì các chỉ số chạy từ `0` đến `d - 1`


<a name="-doc-tung-phan-tu-cua-vector"></a>

### 1.4.3. Đọc từng phần tử của vector
Giả sử:

```python
>>> x = np.array([1, 3, 2])
```

thì thành phần đầu tiên (bằng `1`) được truy cập bằng `x[0]`:

```python
>>> print(x[0])
1
```

Các thành phần tiếp theo được truy cập bằng `x[1]` và `x[2]`, theo thứ tự đó. 

<a name="-chi-so-nguoc"></a>

### 1.4.4. Chỉ số ngược
Trong Python có một điểm đặc biệt là _Chỉ số ngược_. Cho một mảng 1 chiều `x` có `d` phần tử. 
Để truy cập vào phần tử cuối cùng của mảng này, không cần biết `d` là bao nhiêu, ta có thể dùng chỉ số `-1`. 


```python
>>> x = np.array([1, 2, 3])
>>> d = x.shape[0]
>>> print(x[d-1] - x[-1]) 
0
```

Tương tự như thế, phần từ thứ hai từ cuối sẽ được truy cập bằng chỉ số `-2`, ...

**Chú ý:** Nếu một mảng một chiều `x` có số chiều là `d` thì chỉ số `i` trong `x[i]` phải là một số nguyên thoả mãn `-d <= i <= d-1`. Nếu `i` nằm ngoài khoảng này, khi sử dụng `x[i]` sẽ có lỗi `index ... is out of bound...`.

<a name="-thay-doi-gia-tri-mot-phan-tu-cua-mang"></a>

### 1.4.5. Thay đổi giá trị một phần tử của mảng 

Để thay giá trị một phần tử của mảng, ta dùng câu lệnh đơn giản:
```python
>>> a = np.array([1, 2, 3])
>>> a[0] = 4
>>> print(a)
array([4, 2, 3])
```

------- 

**Bài tập:** Thay toàn bộ các phần tử của mảng bằng trung bình cộng các phần tử trong mảng đó, sử dụng vòng `for`. Hàm này không trả về 
biến nào mà chỉ thay đổi các giá trị của biến đầu vào `x`.

-------

<a name="-truy-cap-nhieu-phan-tu-cua-mang-mot-chieu"></a>

## 1.5. Truy cập nhiều phần tử của mảng một chiều 

Để truy cập nhiều phần tử của một mảng một chiều một lúc, chúng ta có nhiều cách khác nhau:

<a name="-doc"></a>

### 1.5.1. Đọc
Ví dụ:
```python
>>> import numpy as np
>>> a = 0.5*np.arange(10)
>>> a
array([ 0. ,  0.5,  1. ,  1.5,  2. ,  2.5,  3. ,  3.5,  4. ,  4.5])
>>> ids = [1, 3, 4, 8]
>>> a[ids]
array([ 0.5,  1.5,  2. ,  4. ])
```
Trong ví dụ này, `ids` là một `list` trong Python, các phần tử của nó đều là các số nguyên nằm trong khoảng `[-10, 9]` nên chúng có thể coi là các chỉ số của mảng `a` được. 
`a[ids]` trả về một mảng numpy, là mảng con của `a` với các phần tử có chỉ số được chỉ ra trong `list` các chỉ số `ids`. 

`ids` cũng có thể là một mảng `numpy` chứa các số nguyên khac là chỉ số hợp lệ của `a`. 
```python
>>> np_ids = np.arange(1, 7, 2) # [1, 3, 5]
array([ 0.5,  1.5,  2.5])
```

Ngoài ra, cách đánh chỉ số của mảng `numpy` cũng sử dụng các quy tắc khác giống như cách đánh chỉ số của một `list`:
```python
>>> a[:3] # return first three elements
array([ 0. ,  0.5,  1. ])
>>> a[-3:] # return last three elements
array([ 3.5,  4. ,  4.5])
>>> a[1:4] # return elements with indexes 1, 2, 3 
array([ 0.5,  1. ,  1.5])
```

<a name="-ghi"></a>

### 1.5.2. Ghi 
Ta cũng có thể thay đổi giá trị của nhiều phần tử trong mảng. Ví dụ: 
```python
>>> a[[1, 3, 5]] = 1 # <=> a[1] = a[3] = a[5] = 1
>>> a
array([ 0. ,  1. ,  1. ,  1. ,  2. ,  1. ,  3. ,  3.5,  4. ,  4.5])
>>> a[-3:] = np.array([0, -1, -2]) # <=> a[-3] = 0, a[-2] = -1, a[-1] = -2
>>> a
array([ 0.,  1.,  1.,  1.,  2.,  1.,  3.,  0., -1., -2.])
>>> a[::2] # return all elements with even indexes 
array([ 0.,  1.,  2.,  3., -1.])
>>> a[::-1] # reverse an array 
array([-2., -1.,  0.,  3.,  1.,  2.,  1.,  1.,  1.,  0.])
```

<a name="-doc-them"></a>

### 1.5.3. Đọc thêm 
[Numpy Indexing and Slicing](https://www.tutorialspoint.com/numpy/numpy_indexing_and_slicing.htm)

----- 

**Bài tập:**
Cho trước một số tự nhiên `n`. Tạo một mảng có `n` phần tử mà các phần tử có chỉ số chẵn (bắt đầu từ 0) là một cấp số cộng bắt đầu từ 2, công sai bằng -0.5; các phần tử có chỉ số lẻ bằng -1.

Ví dụ: 

Với `n=4`, kết quả trả về là mảng `[ 2.  -1.   1.5 -1. ]`.
Với `n=5`, kết quả trả về là mảng `[ 2.  -1.   1.5 -1.   1. ]`.

--------

<a name="-tinh-toan-giua-cac-mang-mot-chieu-va-so-vo-huong"></a>

## 1.6. Tính toán giữa các mảng một chiều và số vô hướng 
<a name="-phep-toan-giua-mang-mot-chieu-voi-mot-so-vo-huong"></a>

### 1.6.1. Phép toán giữa mảng một chiều với một số vô hướng. 

Để cộng/trừ/nhân/chia/luỹ thừa **mọi phần tử** một mảng 1 chiều `x` với một số vô hướng `a` ta chỉ cần lấy `x ? a`, hoặc `a ? x` trong đó `?` có thể thay bằng các phép tính cộng `+`, trừ `-`, nhân `*`, chia `/`, và luỹ thừa `**`. 

```python
>>> x = np.array([1, 2, 3])
>>> a = 3 
>>> x + a
array([4, 5, 6])
>>> 6/x
array([6., 3., 2.])
>>> 3**x
array([3, 9, 27])
```

Chú ý rằng về mặt toán học, **không có phép chia cho vector**. Tuy nhiên, trong Python, ta vẫn hiểu phép chia một số cho một mảng sẽ tương đương với lấy số đó chia cho từng phần tử trong mảng. 

<a name="-phep-toan-giua-hai-mang-mot-chieu"></a>

### 1.6.2. Phép toán giữa hai mảng một chiều

Để có thể tính toán được hai mảng một chiều, số phần tử của hai mảng phải như nhau. Kết quả cũng là một mảng một chiều cùng chiều với hai mảng đó. Các phép toán `+, -, *, /, **` sẽ được thực hiện theo kiểu _element-wise_, tức lấy từng cặp phần tử tương ứng của hai mảng để tính toán rồi lấy kết quả. Ví dụ:

```python
>>> x = np.array([1, 2, 3])
>>> y = np.array([4, 5, 6])
>>> x * y
array([4, 10, 18])
>>> x ** y
array([1, 32, 729])
```

<a name="-cac-ham-toan-hoc"></a>

### 1.6.3. Các hàm toán học 
Các hàm toán học trong numpy như: `np.abs, np.log, np.exp, np.sin, np.cos, np.tan` cũng áp dụng lên từng phần tử của mảng. Hàm `np.log` là logarit tự nhiên, hàm `np.exp` là hàm \\(e^x\\).

```python
>>> x = np.array([1, 2, 3])
>>> np.exp(x)
array([  2.71828183,   7.3890561 ,  20.08553692])
```

Hàm `np.sum(x)` sẽ trả về tổng các phần tử của mảng một chiều `x`. 

-----

**Bài tập:** Cho một mảng 1 chiều `x`, tính mảng `y` và `z` sao cho `y[i] = pi/2 - x[i]` và `z[i] = cos(x[i]) - sin(x[i])`. Sau đó trả về tổng các phần tử của `z`

-----

<a name="-norm-"></a>

## 1.7. Norm 1

Norm 1 của một vector \\(\mathbf{x} \in \mathbb{R}^d\\), kỹ hiệu là \\(\|\|\mathbf{x}\|\|\_1\\), được định nghĩa tổng trị tuyệt đối các phần tử của vector đó: 
\\[
\|\|\mathbf{x}\|\|\_1 = |x_0| + |x_1| + \dots + |x_{d-1}| = \sum_{i = 0}^{d-1} |x_i |
\\]

-----
**Bài tập:**

Viết hàm số tính tổng trị tuyệt đối các phần tử của một mảng một chiều.

(Gợi ý: `np.abs`.)

------

<a name="-ham-softmax-cho-mang-mot-chieu"></a>

## 1.8. Hàm Softmax cho mảng một chiều 
[Softmax Regression](https://machinelearningcoban.com/2017/02/17/softmax/) là một trong số những thuật toán được sử dụng nhiều nhất trong các bài toán Classification. 
Khi triển khai mô hình này, chúng ta cần lập trình hàm softmax. Cho một vector \\(\mathbf{z} \in \mathbb{R}^d\\). Hàm softmax khi áp dụng lên vector \\(\mathbf{z}\\) sẽ 
tạo ra một vector \\(\mathbf{a}\\) cùng chiều với \\(\mathbf{z}\\) và phần tử thứ \\(i\\) (tính từ 0) của nó được xác định bởi: 
\\[
a_i = \frac{\exp(z_i)}{\sum_{j=0}^{d-1} \exp(z_j)}
\\]
với:
\\[
\exp(u) = e^u
\\]
Bạn đọc có thể chứng minh được các phần tử của \\(\mathbf{a}\\) đều nằm trong khoảng \\((0, 1)\\) và có tổng bằng 1. Vì vậy, vector \\(\mathbf{a}\\) còn được coi là vector
xác suất, mỗi phần tử ứng với xác suất của một điểm dữ liệu thuộc vào một class nào đó.


---
**Bài tập:** 
Hãy lập trình hàm softmax.

_Gợi ý:_ Sử dụng hàm `np.exp()`. 

-----

<a name="-tich-vo-huong-cua-hai-vectors---norm-"></a>

## 1.9. Tích vô hướng của hai vectors - Norm 2 
Tích vô hướng (inner product) của hai vectors `x` và `y` có cùng số phần tử được định nghĩa như là: `np.sum(x*y)`, tức lấy `x` nhân với `y` theo element-wise rồi tính tổng các phần tử:
```python
>>> import numpy as np 
>>> x = np.arange(3)
>>> y = np.ones(3)
>>> np.sum(x*y)
3.0
```

Trong numpy, còn hai cách khác để tính tích vô hướng:
```python
>>> x, y = np.arange(3), np.ones(3)
>>> x.dot(y)
3.0
>>> np.dot(x, y)
3.0
```

---
**Bài tập**: Tính norm 2 của một vector - vector này được biểu diễn dưới dạng mảng numpy một chiều. Norm 2 của một vector \\(\mathbf{x}\\), được ký hiệu là \\(\|\|\mathbf{x}\|\|\_2\\), được định nghĩa là căn bậc hai của tổng bình phương các phần tử của nó. 
\\[
\|\|\mathbf{x}\|\|\_2 = \sqrt{x_0^2 + x_1^2 + \dots + x_{d-1}^2}
\\]
trong đó: \\(x_1, \dots, x_{d-1}\\) là các phần tử của vector \\(\mathbf{x} \in \mathbb{R}^d\\). 

Norm 2 được sử dụng rất nhiều trong Machine Learning. Có một hàm khác giúp trực tiếp tính norm, chúng ta sẽ tìm hiểu sau. 

Tìm hiểu thêm: [Norm của vector và ma trận](https://machinelearningcoban.com/math/#-norms-chuan)

---

<a name="-min-max-armin-argmax-cua-mang-mot-chieu"></a>

## 1.10. min, max, armin, argmax của mảng một chiều 
<a name="-min-max"></a>

### 1.10.1. min, max 

Để tìm giá trị lớn nhất hay nhỏ nhất của mảng một chiều, chúng ta đơn giản sử dụng hàm `np.min` hoặc `np.max`. Ví dụ: 
```python
>>> import numpy as np 
>>> a = np.arange(10)
>>> a[-1] = -2 
>>> np.min(a)
-2 
>>> np.max(a)
8
```
hoặc:
```python
>>> a.min()
-2
>>> a.max()
8
```

<a name="-argmin-argmax"></a>

### 1.10.2. argmin, argmax
Để tìm **chỉ số** mà tại đó mảng một chiều đạt giá trị nhỏ nhất hay lớn nhất, ta có thể sử dụng `np.argmin`, hoặc `np.argmax`:

```python
>>> np.argmin(a)
9
>>> np.argmax(a)
8
```
hoặc:
```python
>>> a.argmin()
9
>>> a.argmax()
8
```

------
**Bài tập:**
Trong bài toán classification, sử dụng [Softmax Regression](https://machinelearningcoban.com/2017/02/17/softmax/), giả sử ta đã tính 
được xác suất để một điểm dữ liệu thuộc vào mỗi class. Các xác suất này được lưu dưới dạng một mảng một chiều mà phần tử thứ `i` là 
xác suất để điểm dữ liệu rơi vào lớp `i`. Nhãn của dữ liệu được dự đoán là chỉ số của lớp mà điểm dữ liệu rơi vào với xác suất cao nhất. 
Hãy viết một hàm số xác định chỉ số đó. 

Chú ý: Mảng chứa xác suất này thường được tính bằng cách áp dụng hàm softmax vào _score vector_. Hàm softmax giữ nguyên thứ tự của vector 
đầu vào, vì vậy chỉ số của lớp có xác suất cao nhất cũng là chỉ số của lớp có score cao nhất. (Mời bạn đọc thêm 
bài [Softmax Regression](https://machinelearningcoban.com/2017/02/17/softmax/) để biết thêm chi tiết). 

-----

<a name="softmax-ii---phien-ban-on-dinh"></a>

## Softmax II - Phiên bản ổn định 
Nhắc lại công thức tính softmax của một vector \\(\mathbf{z} \in \mathbb{R}^C\\) (ở đây \\(C\\) là số lượng _lớp_ trong bài toán phân lớp):
\\[
a_i = \frac{\exp(z\_i)}{\sum\_{j=0}^{C-1} \exp(z\_j)}, ~~ \forall i = 0, 1, \dots, C-1\\]

(Xem lại [Hàm Softmax cho mảng một chiều](#/course/5990a766cdc6e32b3b4d0666/lesson/5990aa22cdc6e32b3b4d0667/6).)

Khi một trong các \\(z\_i\\) quá lớn, việc tính toán \\(\exp(z\_i)\\) có thể gây ra hiện tượng tràn số (overflow), ảnh hưởng lớn tới kết quả của hàm softmax. 
Có một cách khắc phục hiện tượng này bằng cách dựa trên quan sát sau: 

\\[
\begin{eqnarray}
\frac{\exp(z_i)}{\sum\_{j=0}^{C-1} \exp(z\_j)} = \frac{\exp(-b)\exp(z\_i)}{\exp(-b)\sum\_{j=0}^{C-1} \exp(z\_j)}
= \frac{\exp(z\_i-b)}{\sum\_{j=0}^{C-1} \exp(z\_j-b)}
\end{eqnarray}
\\]
với \\(b\\) là một hằng số bất kỳ. 

Vậy một phương pháp đơn giản giúp khắc phục hiện tượng overflow là trừ tất cả các \\(z\_i\\) đi một giá trị đủ lớn. 
Trong thực nghiệm, giá trị đủ lớn này thường được chọn là \\(c = \max\_i z\_i\\), tức giá trị lớn nhất của \\(z_i\\).
Vậy chúng ta có thể viết lại hàm `softmax` phía trên bằng cách trừ mỗi phần tử của \\(\mathbf{z}\\) đi giá trị lớn nhất giữa chúng. Ta có phiên bản ổn định hơn được gọi là `softmax_stable`. 

Đọc thêm [Softmax Regression](https://machinelearningcoban.com/2017/02/17/softmax/).
 
-----

**Bài tập:** 
Dựa vào công thức phía trên, hay viết hàm `softmax_stable`, lấy đầu vào là một mảng một chiều (là score vector \\(\mathbf{z}\\)), trả về một mảng một chiều bao gồm toàn bộ các \\(a_i\\) theo công thức.
So sánh kết quả tìm được với kết quả của hàm `softmax` đã thực hiện trước đây.

**Sau khi đã 'Nộp bài' và nhận được kết quả chính xác**, bạn hãy thử làm một thí nghiệm nhỏ dưới đây:
1. Thay dòng `offset = 1000` bằng `offset = 0`, chạy lại chương trình.
2. Sau khi chạy chương trình, cả hai hàm `softmax` và `softmax_stable` vẫn tính được ra kết quả. `nan` ở trên xảy ra vì \\(e^{1000}\\) là một số rất lớn, Python không lưu được số này nên 
trả về _Not a Number_. 

Nói cách khác, hàm `softmax_stable` **stable** hơn hàm `softmax`. 

**Chú ý:** Bạn có thể nhìn thấy thông báo 'Kết quả không chính xác', đừng quan tâm. Đây là câu trả về khi đáp án của bạn khác với đáp án **ban đầu**, tức khi `offset = 1000`.

----

