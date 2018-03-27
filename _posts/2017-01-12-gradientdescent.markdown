---
layout: post
comments: true
title:  "Bài 7: Gradient Descent (phần 1/2)"
title2:  "7. Gradient Descent (1/2)"
date:   2017-01-12 15:22:00
permalink: 2017/01/12/gradientdescent/
mathjax: true
tags: GD Optimization
category: Optimization
sc_project: 11216220
sc_security: 5089fb7c
img: /assets/GD/img2_1.gif 
summary: Gradient Descent một trong những thuật toán quan trọng nhất trong Optimization và Machine Learning 
---
**Trong trang này:**
<!-- MarkdownTOC -->

- [1. Giới thiệu](#-gioi-thieu)
    - [Gradient Descent](#gradient-descent)
- [2. Gradient Descent cho hàm 1 biến](#-gradient-descent-cho-ham--bien)
    - [Ví dụ đơn giản với Python](#vi-du-don-gian-voi-python)
        - [Điểm khởi tạo khác nhau](#diem-khoi-tao-khac-nhau)
        - [Learning rate khác nhau](#learning-rate-khac-nhau)
- [3. Gradient Descent cho hàm nhiều biến](#-gradient-descent-cho-ham-nhieu-bien)
    - [Quay lại với bài toán Linear Regression](#quay-lai-voi-bai-toan-linear-regression)
    - [Sau đây là ví dụ trên Python và một vài lưu ý khi lập trình](#sau-day-la-vi-du-tren-python-va-mot-vai-luu-y-khi-lap-trinh)
        - [Kiểm tra đạo hàm](#kiem-tra-dao-ham)
            - [Giải thích bằng hình học](#giai-thich-bang-hinh-hoc)
            - [Giải thích bằng giải tích](#giai-thich-bang-giai-tich)
            - [Với hàm nhiều biến](#voi-ham-nhieu-bien)
        - [Đường đồng mức \(level sets\)](#duong-dong-muc-level-sets)
- [4. Một ví dụ khác](#-mot-vi-du-khac)
- [5. Thảo luận](#-thao-luan)
- [6. Tài liệu tham khảo](#-tai-lieu-tham-khao)

<!-- /MarkdownTOC -->

<a name="-gioi-thieu"></a>

## 1. Giới thiệu

Các bạn hẳn thấy hình vẽ dưới đây quen thuộc:

<div class="imgcap">
 <img src ="https://github.com/tiepvupsu/tiepvupsu.github.io/blob/master/assets/GD/gradient_descent.png?raw=true" align = "center" width = "600">
</div>

Điểm màu xanh lục là điểm local minimum (cực tiểu), và cũng là điểm làm cho hàm
số đạt giá trị nhỏ nhất. Từ đây trở đi, tôi sẽ dùng _local minimum_ để thay cho
_điểm cực tiểu_, _global minimum_ để thay cho _điểm mà tại đó hàm số đạt giá trị
nhỏ nhất_. Global minimum là một trường hợp đặc biệt của local minimum.

Giả sử chúng ta đang quan tâm đến một hàm số một biến có đạo hàm mọi nơi. Xin
cho tôi được nhắc lại vài điều đã quá quen thuộc:

1. Điểm local minimum \\(x^\*\\) của hàm số là điểm có đạo hàm \\(f'(x^\*)\\)
   bằng 0. Hơn thế nữa, trong lân cận của nó, đạo hàm của các điểm phía bên trái
   \\(x^*\\) là không dương, đạo hàm của các điểm phía bên phải \\(x^\*\\) là
   không âm.

2. Đường tiếp tuyến với đồ thị hàm số đó tại 1 điểm bất kỳ có hệ số góc chính
   bằng đạo hàm của hàm số tại điểm đó.

Trong hình phía trên, các điểm bên trái của điểm local minimum màu xanh lục có
đạo hàm âm, các điểm bên phải có đạo hàm dương. Và đối với hàm số này, càng xa
về phía trái của điểm local minimum thì đạo hàm càng âm, càng xa về phía phải
thì đạo hàm càng dương.

<a name="gradient-descent"></a>

### Gradient Descent
Trong Machine Learning nói riêng và Toán Tối Ưu nói chung, chúng ta thường xuyên
phải tìm giá trị nhỏ nhất (hoặc đôi khi là lớn nhất) của một hàm số nào đó. Ví
dụ như các hàm mất mát trong hai bài [Linear Regression](/2016/12/28/linearregression/) 
và [K-means Clustering](/2017/01/01/kmeans/). Nhìn chung, việc tìm global
minimum của các hàm mất mát trong Machine Learning là rất phức tạp, thậm chí là
bất khả thi. Thay vào đó, người ta thường cố gắng tìm các điểm local minimum, và
ở một mức độ nào đó, coi đó là nghiệm cần tìm của bài toán.

Các điểm local minimum là nghiệm của phương trình đạo hàm bằng 0. Nếu bằng một
cách nào đó có thể tìm được toàn bộ (hữu hạn) các điểm cực tiểu, ta chỉ cần thay
từng điểm local minimum đó vào hàm số rồi tìm điểm làm cho hàm có giá trị nhỏ
nhất (_đoạn này nghe rất quen thuộc, đúng không?_). Tuy nhiên, trong hầu hết các
trường hợp, việc giải phương trình đạo hàm bằng 0 là bất khả thi. Nguyên nhân có
thể đến từ sự phức tạp của dạng của đạo hàm, từ việc các điểm dữ liệu có số
chiều lớn, hoặc từ việc có quá nhiều điểm dữ liệu.

Hướng tiếp cận phổ biến nhất là xuất phát từ một điểm mà chúng ta coi là _gần_
với nghiệm của bài toán, sau đó dùng một phép toán lặp để _tiến dần_ đến điểm
cần tìm, tức đến khi đạo hàm gần với 0. Gradient Descent (viết gọn là GD) và các
biến thể của nó là một trong những phương pháp được dùng nhiều nhất.

<a name="large-scale"></a>

Vì kiến thức về GD khá rộng nên tôi xin phép được chia thành hai phần. Phần 1
này giới thiệu ý tưởng phía sau thuật toán GD và một vài ví dụ đơn giản giúp các
bạn làm quen với thuật toán này và vài khái niệm mới. Phần 2 sẽ nói về các
phương pháp cải tiến GD và các biến thể của GD trong các bài toán mà số chiều và
số điểm dữ liệu lớn. Những bài toán như vậy được gọi là _large-scale_.


<a name="-gradient-descent-cho-ham--bien"></a>

## 2. Gradient Descent cho hàm 1 biến

Quay trở lại hình vẽ ban đầu và một vài quan sát tôi đã nêu. Giả sử
\\(x_\{t}\\) là điểm ta tìm được sau vòng lặp thứ \\(t\\). Ta cần tìm một thuật
toán để đưa \\(x_\{t}\\) về càng gần \\(x^\*\\) càng tốt.

Trong hình đầu tiên, chúng ta lại có thêm hai quan sát nữa:

1. Nếu đạo hàm của hàm số tại \\(x_\{t}\\): \\(f'(x_\{t}) > 0\\) thì
   \\(x_\{t}\\) nằm về bên phải so với \\(x^*\\) (và ngược lại). Để điểm tiếp
   theo \\(x\_{t+1}\\) gần với \\(x^\*\\) hơn, chúng ta cần di chuyển
   \\(x_\{t}\\) về phía bên trái, tức về phía _âm_. Nói các khác, __chúng ta cần
   di chuyển ngược dấu với đạo hàm__:
\\[
x\_{t+1} = x_\{t} + \Delta
\\]
Trong đó \\(\Delta\\) là một đại lượng ngược dấu với đạo hàm \\(f'(x_\{t})\\).

2. \\(x_\{t}\\) càng xa \\(x^*\\) về phía bên phải thì \\(f'(x_\{t})\\) càng lớn
   hơn 0 (và ngược lại). Vậy, lượng di chuyển \\(\Delta\\), một cách trực quan
   nhất, là tỉ lệ thuận với \\(-f'(x_\{t})\\).

Hai nhận xét phía trên cho chúng ta một cách cập nhật đơn giản là:
\\[
x\_{t+1} = x_\{t} - \eta f'(x_\{t})
\\]

Trong đó \\(\eta\\) (đọc là _eta_) là một số dương được gọi là _learning rate_
(tốc độ học). Dấu trừ thể hiện việc chúng ta phải _đi ngược_ với đạo hàm (Đây
cũng chính là lý do phương pháp này được gọi là Gradient Descent - _descent_
nghĩa là _đi ngược_). Các quan sát đơn giản phía trên, mặc dù không phải đúng
cho tất cả các bài toán, là nền tảng cho rất nhiều phương pháp tối ưu nói chung
và thuật toán Machine Learning nói riêng.

<a name="vi-du-don-gian-voi-python"></a>

### Ví dụ đơn giản với Python

Xét hàm số \\(f(x) = x^2 + 5\sin(x)\\) với đạo hàm \\(f'(x) = 2x + 5\cos(x)\\)
(một lý do tôi chọn hàm này vì nó không dễ tìm nghiệm của đạo hàm bằng 0 như hàm
phía trên). Giả sử bắt đầu từ một điểm \\(x\_{0}\\) nào đó, tại vòng lặp thứ
\\(t\\), chúng ta sẽ cập nhật như sau:
\\[
x\_{t+1} = x_\{t} - \eta(2x_\{t} + 5\cos(x_\{t}))
\\]

Như thường lệ, tôi khai báo vài thư viện quen thuộc


```python
# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import math
import numpy as np 
import matplotlib.pyplot as plt
```

Tiếp theo, tôi viết các hàm số :

1. `grad` để tính đạo hàm
2. `cost` để tính giá trị của hàm số. Hàm này không sử dụng trong thuật toán
   nhưng thường được dùng để kiểm tra việc tính đạo hàm của đúng không hoặc để
   xem giá trị của hàm số có giảm theo mỗi vòng lặp hay không.
3. `myGD1` là phần chính thực hiện thuật toán Gradient Desent nêu phía trên. Đầu
   vào của hàm số này là learning rate và điểm bắt đầu. Thuật toán dừng lại khi
   đạo hàm có độ lớn đủ nhỏ.


```python
def grad(x):
    return 2*x+ 5*np.cos(x)

def cost(x):
    return x**2 + 5*np.sin(x)

def myGD1(eta, x0):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    return (x, it)
```


<a name="diem-khoi-tao-khac-nhau"></a>

#### Điểm khởi tạo khác nhau

Sau khi có các hàm cần thiết, tôi thử tìm nghiệm với các điểm khởi tạo khác nhau
là \\(x\_{0} = -5\\) và \\(x\_{0} = 5\\).


```python
(x1, it1) = myGD1(.1, -5)
(x2, it2) = myGD1(.1, 5)
print('Solution x1 = %f, cost = %f, obtained after %d iterations'%(x1[-1], cost(x1[-1]), it1))
print('Solution x2 = %f, cost = %f, obtained after %d iterations'%(x2[-1], cost(x2[-1]), it2))
```

    Solution x1 = -1.110667, cost = -3.246394, obtained after 11 iterations
    Solution x2 = -1.110341, cost = -3.246394, obtained after 29 iterations


Vậy là với các điểm ban đầu khác nhau, thuật toán của chúng ta tìm được nghiệm
gần giống nhau, mặc dù với tốc độ hội tụ khác nhau. Dưới đây là hình ảnh minh
họa thuật toán GD cho bài toán này (_xem tốt trên Desktop ở chế độ full màn
hình_).


<table width = "100%" style = "border: 0px solid white">
   <tr >
        <td width="50%" style = "border: 0px solid white"> 
            <img style="display:block;" width = "100%" src = "/assets/GD/1dimg_5_0.1_-5.gif">
        </td>
        <td width="50%" style = "border: 0px solid white">
            <img style="display:block;" width = "100%" src = "/assets/GD/1dimg_5_0.1_5.gif">
        </td>
    </tr>
</table> 

Từ hình minh họa trên ta thấy rằng ở hình bên trái, tương ứng với \\(x\_{0} = -5\\), nghiệm hội tụ nhanh hơn, vì điểm ban đầu \\(x_0\\) gần với nghiệm \\( x^\* \approx -1\\)  hơn. Hơn nữa, với \\(x\_{0} = 5 \\) ở hình bên phải, _đường đi_ của nghiệm có chứa một khu vực có đạo hàm khá nhỏ gần điểm có hoành độ bằng 2. Điều này khiến cho thuật toán _la cà_ ở đây khá lâu. Khi vượt qua được điểm này thì mọi việc diễn ra rất tốt đẹp.

<a name="learning-rate-khac-nhau"></a>

#### Learning rate khác nhau

Tốc độ hội tụ của GD không những phụ thuộc vào điểm khởi tạo ban đầu mà còn phụ thuộc vào _learning rate_. Dưới đây là một ví dụ với cùng điểm khởi tạo \\(x\_{0} = -5\\) nhưng learning rate khác nhau:

<table width = "100%" style = "border: 0px solid white">
   <tr >
        <td width="50%" style = "border: 0px solid white"> 
            <img style="display:block;" width = "100%" src = "/assets/GD/1dimg_5_0.01_-5.gif">
        </td>
        <td width="50%" style = "border: 0px solid white">
            <img style="display:block;" width = "100%" src = "/assets/GD/1dimg_5_0.5_-5.gif">
        </td>
        <!-- <td width="30%" style = "border: 0px solid white">
            <img src = "/assets/GD/1dimg_5_0.5_5.gif">
        </td> -->
    </tr>
</table> 
Ta quan sát thấy hai điều:

1. Với _learning rate_ nhỏ \\(\eta = 0.01\\), tốc độ hội tụ rất chậm. Trong ví
   dụ này tôi chọn tối đa 100 vòng lặp nên thuật toán dừng lại trước khi tới
   _đích_, mặc dù đã rất gần. Trong thực tế, khi việc tính toán trở nên phức
   tạp, _learning rate_ quá thấp sẽ ảnh hưởng tới tốc độ của thuật toán rất
   nhiều, thậm chí không bao giờ tới được đích.
2. Với _learning rate_ lớn \\(\eta = 0.5\\), thuật toán tiến rất nhanh tới _gần
   đích_ sau vài vòng lặp. Tuy nhiên, thuật toán không hội tụ được vì _bước
   nhảy_ quá lớn, khiến nó cứ _quẩn quanh_ ở đích.

Việc lựa chọn _learning rate_ rất quan trọng trong các bài toán thực tế. Việc
lựa chọn giá trị này phụ thuộc nhiều vào từng bài toán và phải làm một vài thí
nghiệm để chọn ra giá trị tốt nhất. Ngoài ra, tùy vào một số bài toán, GD có thể
làm việc hiệu quả hơn bằng cách chọn ra _learning rate_ phù hợp hoặc chọn
_learning rate_ khác nhau ở mỗi vòng lặp. Tôi sẽ quay lại vấn đề này ở phần 2.

<a name="-gradient-descent-cho-ham-nhieu-bien"></a>

## 3. Gradient Descent cho hàm nhiều biến

Giả sử ta cần tìm global minimum cho hàm \\(f(\mathbf{\theta})\\) trong đó
\\(\mathbf{\theta}\\) (_theta_) là một vector, thường được dùng để ký hiệu tập
hợp các tham số của một mô hình cần tối ưu (trong Linear Regression thì các tham
số chính là hệ số \\(\mathbf{w}\\)). Đạo hàm của hàm số đó tại một điểm
\\(\theta\\) bất kỳ được ký hiệu là \\(\nabla_{\theta}f(\theta)\\) (hình tam
giác ngược đọc là _nabla_). Tương tự như hàm 1 biến, thuật toán GD cho hàm nhiều
biến cũng bắt đầu bằng một điểm dự đoán \\(\theta\_{0}\\), sau đó, ở vòng lặp
thứ \\(t\\), quy tắc cập nhật là:

\\[
\theta\_{t+1} = \theta_\{t} - \eta \nabla_{\theta} f(\theta_\{t})
\\]

Hoặc viết dưới dạng đơn giản hơn: \\(\theta = \theta - \eta \nabla_{\theta} f(\theta)\\).

Quy tắc cần nhớ: **luôn luôn đi ngược hướng với đạo hàm**.

Việc tính toán đạo hàm của các hàm nhiều biến là một kỹ năng cần thiết. Một vài đạo hàm đơn giản có thể được [tìm thấy ở đây](/math/#bang-cac-dao-ham-co-ban).
<a name="quay-lai-voi-bai-toan-linear-regression"></a>

### Quay lại với bài toán Linear Regression
Trong mục này, chúng ta quay lại với bài toán [Linear Regression](/2016/12/28/linearregression/) và thử tối ưu hàm mất mát của nó bằng thuật toán GD. 

Hàm mất mát của Linear Regression là: 
\\[
\mathcal{L}(\mathbf{w}) = \frac{1}{2N}\|\|\mathbf{y - \bar{X}w}\|\|_2^2
\\]

**Chú ý**: hàm này có khác một chút so với hàm tôi nêu trong bài [Linear Regression](/2016/12/28/linearregression/). Mẫu số có thêm \\(N\\) là số lượng dữ liệu trong training set. Việc lấy trung bình cộng của lỗi này nhằm giúp tránh trường hợp hàm mất mát và đạo hàm có giá trị là một số rất lớn, ảnh hưởng tới độ chính xác của các phép toán khi thực hiện trên máy tính. Về mặt toán học, nghiệm của hai bài toán là như nhau.

Đạo hàm của hàm mất mát là:
\\[
\nabla_{\mathbf{w}}\mathcal{L}(\mathbf{w}) = 
\frac{1}{N}\mathbf{\bar{X}}^T \mathbf{(\bar{X}w - y)} ~~~~~(1)
\\]

<a name="sau-day-la-vi-du-tren-python-va-mot-vai-luu-y-khi-lap-trinh"></a>

### Sau đây là ví dụ trên Python và một vài lưu ý khi lập trình

Load thư viện


```python
# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
np.random.seed(2)
```

Tiếp theo, chúng ta tạo 1000 điểm dữ liệu được chọn _gần_ với đường thẳng \\(y = 4 + 3x\\), hiển thị chúng và tìm nghiệm theo công thức:


```python
X = np.random.rand(1000, 1)
y = 4 + 3 * X + .2*np.random.randn(1000, 1) # noise added

# Building Xbar 
one = np.ones((X.shape[0],1))
Xbar = np.concatenate((one, X), axis = 1)

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w_lr = np.dot(np.linalg.pinv(A), b)
print('Solution found by formula: w = ',w_lr.T)

# Display result
w = w_lr
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(0, 1, 2, endpoint=True)
y0 = w_0 + w_1*x0

# Draw the fitting line 
plt.plot(X.T, y.T, 'b.')     # data 
plt.plot(x0, y0, 'y', linewidth = 2)   # the fitting line
plt.axis([0, 1, 0, 10])
plt.show()
```

    Solution found by formula: w =  [[ 4.00305242  2.99862665]]


<div class="imgcap">
 <img src ="/assets/GD/output_11_1.png" align = "center" width = "400">
</div>


Đường thẳng tìm được là đường có màu vàng có phương trình \\(y \approx 4 + 2.998x\\).

Tiếp theo ta viết đạo hàm và hàm mất mát:


```python
def grad(w):
    N = Xbar.shape[0]
    return 1/N * Xbar.T.dot(Xbar.dot(w) - y)

def cost(w):
    N = Xbar.shape[0]
    return .5/N*np.linalg.norm(y - Xbar.dot(w), 2)**2;
```

<a name="kiem-tra-dao-ham"></a>

#### Kiểm tra đạo hàm
Việc tính đạo hàm của hàm nhiều biến thông thường khá phức tạp và rất dễ mắc lỗi, nếu chúng ta tính sai đạo hàm thì thuật toán GD không thể chạy đúng được. Trong thực nghiệm, có một cách để kiểm tra liệu đạo hàm tính được có chính xác không. Cách này dựa trên định nghĩa của đạo hàm (cho hàm 1 biến):
\\[
f'(x) = \lim_{\varepsilon \rightarrow 0}\frac{f(x + \varepsilon) - f(x)}{\varepsilon}
\\]

Một cách thường được sử dụng là lấy một giá trị \\(\varepsilon \\) rất nhỏ, ví dụ \\(10^{-6}\\), và sử dụng công thức:
\\[
f'(x) \approx \frac{f(x + \varepsilon) - f(x - \varepsilon)}{2\varepsilon} ~~~~ (2)
\\]

Cách tính này được gọi là _numerical gradient_.

**Câu hỏi: Tại sao công thức xấp xỉ hai phía trên đây lại được sử dụng rộng rãi, sao không sử dụng công thức xấp xỉ đạo hàm bên phải hoặc bên trái?**

Có hai các giải thích cho vấn đề này, một bằng hình học, một bằng giải tích. 

<a name="giai-thich-bang-hinh-hoc"></a>

##### Giải thích bằng hình học 
Quan sát hình dưới đây:

<div class="imgcap">
 <img src ="/assets/GD/check_grad.png" align = "center" width = "600">
</div>
<br>
Trong hình, vector màu đỏ là đạo hàm *chính xác* của hàm số tại điểm có hoành độ bằng \\(x_0\\). Vector màu xanh lam (có vẻ là hơi tím sau khi convert từ .pdf sang .png) thể hiện cách xấp xỉ đạo hàm phía phải. Vector màu xanh lục thể hiện cách xấp xỉ đạo hàm phía trái. Vector màu nâu thể hiện cách xấp xỉ đạo hàm hai phía. Trong ba vector xấp xỉ đó, vector xấp xỉ hai phía màu nâu là gần với vector đỏ nhất nếu xét theo hướng. 

Sự khác biệt giữa các cách xấp xỉ còn lớn hơn nữa nếu tại điểm x, hàm số bị *bẻ cong* mạnh hơn. Khi đó, xấp xỉ trái và phải sẽ khác nhau rất nhiều. Xấp xỉ hai bên sẽ *ổn định* hơn.

<a name="giai-thich-bang-giai-tich"></a>

##### Giải thích bằng giải tích
Chúng ta cùng quay lại một chút với Giải tích I năm thứ nhất đại học: [Khai triển Taylor](http://mathworld.wolfram.com/TaylorSeries.html). 

Với \\(\varepsilon\\) rất nhỏ, ta có hai xấp xỉ sau: 

\\[
f(x + \varepsilon) \approx f(x) + f'(x)\varepsilon + \frac{f"(x)}{2} \varepsilon^2 + \dots
\\]

và:
\\[
f(x - \varepsilon) \approx f(x) - f'(x)\varepsilon + \frac{f"(x)}{2} \varepsilon^2 - \dots
\\]

Từ đó ta có: 
\\[
\frac{f(x + \varepsilon) - f(x)}{\varepsilon} \approx f'(x) + \frac{f"(x)}{2}\varepsilon + \dots =  f'(x) + O(\varepsilon) ~~ (3)
\\]

\\[
\frac{f(x + \varepsilon) - f(x - \varepsilon)}{2\varepsilon} \approx f'(x) + \frac{f^{(3)}(x)}{6}\varepsilon^2 + \dots =  f'(x) + O(\varepsilon^2) ~~(4)
\\]


Từ đó, nếu xấp xỉ đạo hàm bằng công thức \\((3)\\) (xấp xỉ đạo hàm phải), sai số sẽ là \\(O(\varepsilon)\\). Trong khi đó, nếu xấp xỉ đạo hàm bằng công thức \\((4)\\) (xấp xỉ đạo hàm hai phía), sai số sẽ là \\(O(\varepsilon^2) \ll O(\varepsilon)\\) nếu \\(\varepsilon\\) nhỏ. 

Cả hai cách giải thích trên đây đều cho chúng ta thấy rằng, xấp xỉ đạo hàm hai
phía là xấp xỉ tốt hơn.

<a name="voi-ham-nhieu-bien"></a>

##### Với hàm nhiều biến

Với hàm nhiều biến, công thức \\((2)\\) được áp dụng cho từng biến khi các biến
khác cố định. Cách tính này thường cho giá trị khá chính xác. Tuy nhiên, cách
này không được sử dụng để tính đạo hàm vì độ phức tạp quá cao so với cách tính
trực tiếp. Khi so sánh đạo hàm này với đạo hàm chính xác tính theo công thức,
người ta thường giảm số chiều dữ liệu và giảm số điểm dữ liệu để thuận tiện cho
tính toán. Một khi đạo hàm tính được rất gần với _numerical gradient_, chúng ta
có thể tự tin rằng đạo hàm tính được là chính xác.

Dưới đây là một đoạn code đơn giản để kiểm tra đạo hàm và có thể áp dụng với một
hàm số (của một vector) bất kỳ với `cost` và `grad` đã tính ở phía trên.


```python
def numerical_grad(w, cost):
    eps = 1e-4
    g = np.zeros_like(w)
    for i in range(len(w)):
        w_p = w.copy()
        w_n = w.copy()
        w_p[i] += eps 
        w_n[i] -= eps
        g[i] = (cost(w_p) - cost(w_n))/(2*eps)
    return g 

def check_grad(w, cost, grad):
    w = np.random.rand(w.shape[0], w.shape[1])
    grad1 = grad(w)
    grad2 = numerical_grad(w, cost)
    return True if np.linalg.norm(grad1 - grad2) < 1e-6 else False 

print( 'Checking gradient...', check_grad(np.random.rand(2, 1), cost, grad))
```

    Checking gradient... True

(_Với các hàm số khác, bạn đọc chỉ cần viết lại hàm `grad` và `cost` ở phần trên
rồi áp dụng đoạn code này để kiểm tra đạo hàm. Nếu hàm số là hàm của một ma trận
thì chúng ta thay đổi một chút trong hàm `numerical_grad`, tôi hy vọng không quá
phức tạp_).

Với bài toán Linear Regression, cách tính đạo hàm như trong \\((1)\\) phía trên
được coi là đúng vì sai số giữa hai cách tính là rất nhỏ (nhỏ hơn
\\(10^{-6}\\)). Sau khi có được đạo hàm chính xác, chúng ta viết hàm cho GD:


```python
def myGD(w_init, grad, eta):
    w = [w_init]
    for it in range(100):
        w_new = w[-1] - eta*grad(w[-1])
        if np.linalg.norm(grad(w_new))/len(w_new) < 1e-3:
            break 
        w.append(w_new)
    return (w, it) 

w_init = np.array([[2], [1]])
(w1, it1) = myGD(w_init, grad, 1)
print('Solution found by GD: w = ', w1[-1].T, ',\nafter %d iterations.' %(it1+1))
```

    Solution found by GD: w =  [[ 4.01780793  2.97133693]] ,
    after 49 iterations.


Sau 49 vòng lặp, thuật toán đã hội tụ với một nghiệm khá gần với nghiệm tìm được
theo công thức.

Dưới đây là hình động minh họa thuật toán GD.



<table width = "100%" style = "border: 0px solid white">
   <tr >
        <td width="40%" style = "border: 0px solid white"> 
        <img style="display:block;" width = "100%" src = "/assets/GD/img1_1.gif">
         </td>
        <td width="40%" style = "border: 0px solid white">
        <img style="display:block;" width = "100%" src = "/assets/GD/img2_1.gif">
        </td>
    </tr>
</table> 

Trong hình bên trái, các đường thẳng màu đỏ là nghiệm tìm được sau mỗi vòng lặp. 

Trong hình bên phải, tôi xin giới thiệu một thuật ngữ mới: _đường đồng mức_.
<a name="duong-dong-muc-level-sets"></a>

#### Đường đồng mức (level sets)

Với đồ thị của một hàm số với hai biến đầu vào cần được vẽ trong không gian ba
chiều, nhều khi chúng ta khó nhìn được nghiệm có khoảng tọa độ bao nhiêu. Trong
toán tối ưu, người ta thường dùng một cách vẽ sử dụng khái niệm _đường đồng mức_
(level sets).

Nếu các bạn để ý trong các bản độ tự nhiên, để miêu tả độ cao của các dãy núi,
người ta dùng nhiều đường cong kín bao quanh nhau như sau:

<div class="imgcap">
 <img src ="http://files.vforum.vn/2016/T06/img/vforum.vn-324944-hinh-44-lc6b0e1bba3c-c491e1bb93-c491e1bb8ba-hc3acnh-te1bb89-le1bb87-le1bb9bn.png" align = "center" width = "600">
 <div class = "thecap"> Ví dụ về đường đồng mức trong các bản đồ tự nhiên. (Nguồn: <a href = "http://vforum.vn/diendan/showthread.php?90166-Dia-ly-6-Duong-dong-muc-la-nhung-duong-nhu-the-nao-">Địa lý 6: Đường đồng mức là những đường như thế nào?</a>)</div>
</div>


Các vòng nhỏ màu đỏ hơn thể hiện các điểm ở trên cao hơn. 

Trong toán tối ưu, người ta cũng dùng phương pháp này để thể hiện các bề mặt
trong không gian hai chiều.

Quay trở lại với hình minh họa thuật toán GD cho bài toán Liner Regression bên
trên, hình bên phải là hình biểu diễn các level sets. Tức là tại các điểm trên
cùng một vòng, hàm mất mát có giá trị như nhau. Trong ví dụ này, tôi hiển thị
giá trị của hàm số tại một số vòng. Các vòng màu xanh có giá trị thấp, các vòng
tròn màu đỏ phía ngoài có giá trị cao hơn. Điểm này khác một chút so với đường
đồng mức trong tự nhiên là các vòng bên trong thường thể hiện một thung lũng hơn
là một đỉnh núi (vì chúng ta đang đi tìm giá trị nhỏ nhất).

Tôi thử với _learning rate_ nhỏ hơn, kết quả như sau:

<table width = "100%" style = "border: 0px solid white">
   <tr >
        <td width="40%" style = "border: 0px solid white"> 
        <img style="display:block;" width = "100%" src = "/assets/GD/img1_0.1.gif">
         </td>
        <td width="40%" style = "border: 0px solid white">
        <img style="display:block;" width = "100%" src = "/assets/GD/img2_0.1.gif">
        </td>
    </tr>
</table> 

Tốc độ hội tụ đã chậm đi nhiều, thậm chí sau 99 vòng lặp, GD vẫn chưa tới gần
được nghiệm tốt nhất. Trong các bài toán thực tế, chúng ta cần nhiều vòng lặp
hơn 99 rất nhiều, vì số chiều và số điểm dữ liệu thường là rất lớn.

<a name="-mot-vi-du-khac"></a>

## 4. Một ví dụ khác

Để kết thúc phần 1 của Gradient Descent, tôi xin nêu thêm một ví dụ khác.

<div class="imgcap">
 <img src ="/assets/GD/img3_0.015.gif" align = "center" width = "800">
</div>

Hàm số \\(f(x, y) = (x^2 + y - 7)^2 + (x - y + 1)^2\\) có hai điểm local minimum
màu xanh lục tại \\((2, 3)\\) và \\((-3, -2)\\), và chúng cũng là hai điểm
global minimum. Trong ví dụ này, tùy vào điểm khởi tạo mà chúng ta thu được các
nghiệm cuối cùng khác nhau.

<a name="-thao-luan"></a>

## 5. Thảo luận

Dựa trên GD, có rất nhiều thuật toán phức tạp và hiệu quả hơn được thiết kế cho
những loại bài toán khác nhau. Vì bài này đã đủ dài, tôi xin phép dừng lại ở
đây. Mời các bạn đón đọc bài Gradient Descent phần 2 với nhiều kỹ thuật nâng cao
hơn.

<a name="-tai-lieu-tham-khao"></a>

## 6. Tài liệu tham khảo
1. [An overview of gradient descent optimization algorithms](http://sebastianruder.com/optimizing-gradient-descent/)
2. [An Interactive Tutorial on Numerical Optimization](http://www.benfrederickson.com/numerical-optimization/)
3. [Gradient Descent by Andrew NG](https://www.youtube.com/watch?v=eikJboPQDT0)
