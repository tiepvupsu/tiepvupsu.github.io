---
layout: post
title: Frequently Asked Questions
comments: true 
permalink: /faqs/
---
<!-- MarkdownTOC depth=2 -->

- [1. Blog được tạo như thế nào?](#-blog-duoc-tao-nhu-the-nao)
- [2. Các kiến thức cần thiết để học Machine Learning?](#-cac-kien-thuc-can-thiet-de-hoc-machine-learning)
- [3. Hướng dẫn cài đặt python và các thư viện trên MacOS?](#-huong-dan-cai-dat-python-va-cac-thu-vien-tren-macos)
- [4. Hướng dẫn cài đặt python và các thư viện trên Windows?](#-huong-dan-cai-dat-python-va-cac-thu-vien-tren-windows)

<!-- /MarkdownTOC -->


<a name="-blog-duoc-tao-nhu-the-nao"></a>

## 1. Blog được tạo như thế nào?
Mới bạn đọc bài [Blog và các bài viết được tạo như thế nào](/2017/02/02/howdoIcreatethisblog/) để hiểu rõ hơn. 

<a name="-cac-kien-thuc-can-thiet-de-hoc-machine-learning"></a>

## 2. Các kiến thức cần thiết để học Machine Learning? 

* **Toán**: bạn cần nắm vững các môn Giải Tích, Đại Số Tuyến Tính, Xác Suất Thông Kê. Nếu biết thêm về Toán Tối Ưu nữa thì rất tốt. 

* **Lập trình**: bạn có thể dùng Matlab, Python, R. Hiện tại tôi thấy Python được sử dụng rộng rãi nhất vì có nhiều thư viện hỗ trợ, và cũng free nữa. 

* **Tiếng Anh**: ít nhất là kỹ năng đọc, các bạn nên trau dồi càng sớm càng tốt. Đọc dần các tài liệu tiếng Anh để luyện cả tiếng và đọc thêm kiến thức mới. 

[Về đầu trang](/faqs/).

<a name="-huong-dan-cai-dat-python-va-cac-thu-vien-tren-macos"></a>

## 3. Hướng dẫn cài đặt python và các thư viện trên MacOS?
**Xin cảm ơn facebook Nguyễn Nghĩa về phần hướng dẫn này:**

Nhận thấy có một số bạn gặp khó khắn trong việc sử dụng source code trong các bài viết về Machine Learning cơ bản như việc cài đặt thư viện, import các thư viện đó để có thể chạy được những source trong các bài viết. Trong bài viết hôm nay tôi sẽ hướng dẫn các bạn cách cài đặt một số thư viện trên hệ điều hành MacOS nhằm phục vụ cho mục đích tìm hiểu và nghiên cứu về Machine Learning.

<a name="--doi-net-ve-thu-vien-scikit-learn"></a>

### 3. 1. Đôi nét về thư viện scikit-learn
Theo như tôi biết thì Scikit-learn mà một trong những thư viện mã nguồn mở Machine Learning viết bằng Python và được đông đảo mọi người sử dụng nhất hiện nay. Scikit-learn implement nhiều thuật toán máy học từ các thuật toán cơ bản cho đến các thuật toán phức tạp như DecisonTree, Naive Bayes, K-Nearest Neighbor (KNN), Support Vector Machine (SVM), Artificial Nerual Network (ANN)...

Trang chủ của thư viện: http://scikit-learn.org/

<a name="-cai-dat-thu-vien-scikit-learn"></a>

### 3.2. Cài đặt thư viện scikit-learn
Thư viện scikit-learn yêu cầu chúng ta phải cài đặt những module như dưới đây:
* Python (>= 2.6 or >= 3.3),
* NumPy (>= 1.6.1),
* SciPy (>= 0.9).

<a name="python"></a>

### 3.3  Python
Phiên bản mới nhất của hệ điều hành macOS (Sierra) thì python 2.7 đã được cài đặt sẵn, vì vậy chúng ta không cài đặt lại python. Với những verson khác thì trước khi cài đặt python chúng ta mở **Termial**  và gõ lệnh gõ lệnh python để kiểm tra python đã đượcc cài đặt hay chưa.

```
 python
```

Nếu python chưa được cài đặt thì sẽ xuất ra thông báo lỗi và chúng ta sử dụng lệnh dưới đây để cài đặt.
```
brew install python
```

Ngược lại sẽ xuất ra thông tin chi tiết về phiên bản python đang sử dụng và đi vào môi trường lập trình python.

Nếu bạn nào muốn cài đặt python phiên bản 3.6 thì sử dụng lệnh

```
brew install python3
```

<a name="pip"></a>

### 3.4  Pip
Pip là một công cụ nhỏ gọn giúp chúng ta cài đặt các gói thư viện trong pytho một cách nhanh chóng. Hầu hết mọi thư viện của python đều được cài đặt qua pip. Và để cài đặt được pip chúng ta sử dụng lệnh:
```
sudo easy_install pip
```

<a name="numpy"></a>

### 3.5. Numpy

```
pip install numpy
```

<a name="scipy"></a>

### 3.6. Scipy

```
pip install scipy
```

<a name="matplotlib"></a>

### 3.7. Matplotlib
Matplotlib là một thư viện python phục vụ cho việc vẽ đồ thị. Lệnh cài đặt matplotlib

```
pip install matplotlib
```

<a name="scikit-learn"></a>

### 3.8. Scikit-learn
Sau khi đã cài đặt xong các module mà scikit-learn yêu cầu thì chúng ta sử dụng lệnh dưới đây để cài đặt thư viện Machine Learning này.

```
pip install -U scikit-learn
```

Nếu gặp lỗi về permission thì các bạn sử dụng lệnh dưới

```
sudo pip install -U scikit-learn
```

<a name="kiem-tra-cai-dat"></a>

### 3.9 Kiểm tra cài đặt
Sau khi cài đặt đẩy đủ các thư viện thì bước cuối cùng sẽ là thử **import** thư viện để kiểm tra lại quá trình cài đặt có thành công hay không. Hãy thư vào một trường lập trình python bằng cách mở Terminal và gõ lệnh:

```
python
```

Thử import thư viện bằng lệnh sau:
```python
import sklearn
```

Nếu không có thông báo nào nghĩa là chúng ta đã install thành công. Ngược lại các bạn phải quay lại cài đặt các thư viện ở các bước trên.

<a name="-tai-lieu-tham-khao"></a>

### 3.10. Tài liệu tham khảo
[1] [Installing scikit-learn](http://scikit-learn.org/stable/install.html)

[Về đầu trang](/faqs/).

<a name="-huong-dan-cai-dat-python-va-cac-thu-vien-tren-windows"></a>

## 4. Hướng dẫn cài đặt python và các thư viện trên Windows?

**Cảm ơn facebook Pham Chi Hieu về phần trả lời này:**

[Bài: Cài đặt Python và thư viện sử dụng Anaconda trên Windows](https://chieupham.github.io/2017/02/18/Python-Windows/)










