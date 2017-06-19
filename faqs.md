---
layout: post
title: Frequently Asked Questions
comments: true 
permalink: /faqs/
---
<!-- MarkdownTOC depth=1 -->

- [1. Blog được tạo như thế nào?](#-blog-duoc-tao-nhu-the-nao)
- [2. Các kiến thức cần thiết để học Machine Learning?](#-cac-kien-thuc-can-thiet-de-hoc-machine-learning)
- [3. Hướng dẫn cài đặt python và các thư viện trên MacOS?](#-huong-dan-cai-dat-python-va-cac-thu-vien-tren-macos)
- [4. Hướng dẫn cài đặt python và các thư viện trên Windows?](#-huong-dan-cai-dat-python-va-cac-thu-vien-tren-windows)
- [5. Các sách tham khảo?](#-cac-sach-tham-khao)
- [6. Làm thế nào để hỗ trợ blog](#-lam-the-nao-de-ho-tro-blog)

<!-- /MarkdownTOC -->

**Bạn cũng có thể đặt các câu hỏi và tham gia thảo luận tại:**
[**Forum Machine Learning cơ bản**](https://www.facebook.com/groups/257768141347267/)



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

<a name="--python"></a>

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

<a name="--pip"></a>

### 3.4  Pip
Pip là một công cụ nhỏ gọn giúp chúng ta cài đặt các gói thư viện trong pytho một cách nhanh chóng. Hầu hết mọi thư viện của python đều được cài đặt qua pip. Và để cài đặt được pip chúng ta sử dụng lệnh:
```
sudo easy_install pip
```

<a name="-numpy"></a>

### 3.5. Numpy

```
pip install numpy
```

<a name="-scipy"></a>

### 3.6. Scipy

```
pip install scipy
```

<a name="-matplotlib"></a>

### 3.7. Matplotlib
Matplotlib là một thư viện python phục vụ cho việc vẽ đồ thị. Lệnh cài đặt matplotlib

```
pip install matplotlib
```

<a name="-scikit-learn"></a>

### 3.8. Scikit-learn
Sau khi đã cài đặt xong các module mà scikit-learn yêu cầu thì chúng ta sử dụng lệnh dưới đây để cài đặt thư viện Machine Learning này.

```
pip install -U scikit-learn
```

Nếu gặp lỗi về permission thì các bạn sử dụng lệnh dưới

```
sudo pip install -U scikit-learn
```

<a name="-kiem-tra-cai-dat"></a>

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


<a name="-cai-dat-python-bang-anaconda"></a>

### 4.1. Cài đặt Python bằng Anaconda.
Để tải về Python và một số thư viện cần thiết, một cách đơn giản nhất là tải về [Anaconda cho windows](https://docs.continuum.io/anaconda/install#anaconda-for-windows-install/) và cài đặt vào thư mục bạn muốn. Anaconda hỗ trợ rất nhiều thư viện giúp lập trình Python. 

Sau khi cài đặt xong, bạn vào thư mục Scripts trong thư mục Anaconda vừa cài đặt, và khởi động Spyder. Các bạn có thể sử dụng môi trường IDE nào cũng được. Tôi hay sử dụng Spyder vì layout của nó khá giống với Matlab, chúng ta có thể quan sát được Script, Console và các biến. Console cho Python của Spyder bao gồm Python hoặc IPython notebook.

<div class="imgcap">
<img src ="/assets/images/spyder.PNG" width = "700" align = "center">
<div class="thecap"> Giao diện Spyder trên Windows. <br></div>
</div>

<a name="-kiem-tra-libs"></a>

### 4.2. Kiểm tra Libs
Anaconda đã có sẵn khá là nhiều thư viện python như : [Numpy](http://www.numpy.org/), [Scipy](https://www.scipy.org/), [Matplotlib](http://matplotlib.org/) , [sklearn](http://scikit-learn.org/stable/)

Để kiểm tra python của Anaconda đã có thư viện nào đó, chúng ta sẽ thử import nó trong Console.

```python
>>> import numpy
```

Không có lỗi được thông báo nghĩa là python đã biết được thư viện này. Để kiểm tra thư viện này ở đâu, sau khi *import*, ta truy xuất đường dẫn của thư viện như sau:

```python
>>> numpy.__file__
'C:\\These\\soft\\Anaconda2\\lib\\site-packages\\numpy\\__init__.pyc'
```

Thư viện Numpy của tôi nằm ở đường dẫn *'C:\\These\\soft\\Anaconda2\\lib\\site-packages\\'* . Anaconda đã có sẵn thư viện Numpy

```python
>>> import sklearn
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ImportError: No module named sklearn
```
Nếu như Python trả về lỗi Import như trên thì có nghĩa trong Anaconda chúng ta chưa có thư viện đó.

<a name="-cai-dat-libs-bang-anaconda"></a>

### 4.3. Cài đặt Libs bằng Anaconda
Ở phần trên python của tôi chưa có thư viện *sklearn*, nên tôi phải đi cài đặt nó. Vì tôi sử dụng Anaconda cho lập trình python nên tôi cần phải *(1) cài đặt thư viện mới vào đường dẫn libs python của Anaconda* hoặc *(2) chỉ cho python của Anaconda biết về đường dẫn tới thư viện mới này*.

Với Anaconda, việc cài đặt 1 thư viện đang được hỗ trợ cực kỳ đơn giản, tôi chỉ cần dùng tools *pip* hoặc *conda* mà Anaconda đã cài sẵn. Cụ thể, ở đây tôi muốn cài thư viện sklearn tôi truy cập vào trang chủ của [sklearn](http://scikit-learn.org/stable/install.html). Trang này ghi rằng chúng ta có thể cài bằng *pip* hoặc *conda*.

Chúng ta sẽ bật cmd (Command Prompt) của windows lên và gõ lệnh *conda install scikit-learn* hoặc *pip install -U scikit-learn*. Conda sẽ tự động tìm thư viện *sklearn* và cài vào đường dẫn Anaconda giúp chúng ta.

```python
C:>conda install scikit-learn
```

<div class="imgcap">
<img src ="/assets/images/cmd_conda.png" width = "600" align = "center">
<div class="thecap"> Sử dụng conda qua cmd của windows. <br></div>
</div>

Chờ cho thư viện và các thư viện liên quan hoàn tất cài đặt, chúng ta vào spyder kiểm tra lại đã có *sklearn* chưa. Và python trả về đã có sklearn trong Anaconda. Và chúng ta đã có thể sử dụng sklearn.

```python
>>> import sklearn
>>> sklearn.__file__
'C:\\These\\soft\\Anaconda2\\lib\\site-packages\\sklearn\\__init__.pyc'
>>> 
```

Với 1 thư viện chưa có trên Anaconda, cách cài đặt sẽ phức tạp hơn chút nhưng hầu hết các thư viện lớn thường dùng đều có thể cài đặt thông qua Anaconda, nên chúng ta không phải lo lắng lắm. Để cài loại thư viện như vậy, tôi sẽ chỉ dẫn vào những bài sau.

<a name="-chay-thu--doan-code-tren-python"></a>

### 4.4. Chạy thử 1 đoạn code trên python.

Bây giờ, các bạn đã có thể chạy thử 1 vài ví dụ trên trang Machine Learning cơ bản, ví dụ như [Bài 3: Linear Regression](/2016/12/28/linearregression/)



[Về đầu trang](/faqs/).

<a name="-cac-sach-tham-khao"></a>

## 5. Các sách tham khảo?

Mời bạn [xem tại đây](/2017/02/02/howdoIcreatethisblog/#main-references).

<a name="-lam-the-nao-de-ho-tro-blog"></a>

## 6. Làm thế nào để hỗ trợ blog 

_Nội dung trên blog này là hoàn toàn miễn phí. Tôi cũng không sử dụng dịch vụ quảng cáo nào vì không muốn làm phiền các bạn trong khi đọc. Tuy nhiên, nếu bạn thấy nội dung blog hữu ích và muốn ủng hộ blog, bạn có thể **Mời tôi một ly cà phê** bằng cách click vào nút 'Buy me a coffee' ở phía trên cột bên trái của blog, loại cà phê mà bạn vẫn thích uống :). Tôi xin chân thành cảm ơn._

[Về đầu trang](/faqs/).
