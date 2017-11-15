---
layout: post
comments: true
title:  "Bài 15: Overfitting"
title2:  "15. Overfitting"
date:   2017-03-04 15:22:00
permalink: 2017/03/04/overfitting/
mathjax: true
tags: Overfitting
category: General
sc_project: 11274171
sc_security: 980b6518
img: \assets\15_overfitting\nnet_reg0.png
summary: Overfitting không phải là một thuật toán trong Machine Learning. Nó là một hiện tượng không mong muốn thường gặp, người xây dựng mô hình Machine Learning cần nắm được các kỹ thuật để tránh hiện tượng này.
---

**Trong trang này:**
<!-- MarkdownTOC -->

- [1. Giới thiệu](#-gioi-thieu)
- [2. Validation](#-validation)
    - [2.1. Validation](#-validation-1)
    - [2.2. Cross-validation](#-cross-validation)
- [3. Regularization](#-regularization)
    - [3.1. Early Stopping](#-early-stopping)
    - [3.2. Thêm số hạng vào hàm mất mát](#-them-so-hang-vao-ham-mat-mat)
    - [3.3. \\\(l_2\\\) regularization](#-%5C%5Cl%5C%5C-regularization)
        - [Ví dụ về Weight Decay với MLP](#vi-du-ve-weight-decay-voi-mlp)
    - [3.4. Tikhonov regularization](#-tikhonov-regularization)
    - [3.5. Regularizers for sparsity](#-regularizers-for-sparsity)
    - [3.6. Regularization trong sklearn](#-regularization-trong-sklearn)
- [4. Các phương pháp khác](#-cac-phuong-phap-khac)
- [5. Tóm tắt nội dung](#-tom-tat-noi-dung)
- [6. Tài liệu tham khảo](#-tai-lieu-tham-khao)

<!-- /MarkdownTOC -->


Overfitting không phải là một thuật toán trong Machine Learning. Nó là một hiện tượng không mong muốn thường gặp, người xây dựng mô hình Machine Learning cần nắm được các kỹ thuật để tránh hiện tượng này.

<a name="-gioi-thieu"></a>

<a name="-gioi-thieu"></a>
## 1. Giới thiệu
Đây là một câu chuyện của chính tôi khi lần đầu biết đến Machine Learning.

Năm thứ ba đại học, một thầy giáo có giới thiệu với lớp tôi về Neural Networks. Lần đầu tiên nghe thấy khái niệm này, chúng tôi hỏi thầy mục đích của nó là gì. Thầy nói, về cơ bản, từ dữ liệu cho trước, chúng ta cần tìm một hàm số để biến các các điểm đầu vào thành các điểm đầu ra tương ứng, không cần chính xác, chỉ cần xấp xỉ thôi.

Lúc đó, vốn là một học sinh chuyên toán, làm việc nhiều với đa thức ngày cấp ba, tôi đã quá tự tin trả lời ngay rằng [Đa thức Nội suy Lagrange](http://vuontoanblog.blogspot.com/2012/10/polynomial-interpolation-lagrange.html) có thể làm được điều đó, miễn là các điểm đầu vào khác nhau đôi một! Thầy nói rằng "những gì ta biết chỉ là nhỏ xíu so với những gì ta chưa biết". Và đó là những gì tôi muốn bắt đầu trong bài viết này.

Nhắc lại một chút về Đa thức nội suy Lagrange: Với \\(N\\) cặp điểm dữ liệu \\((x_1, y_1), (x_2, y_2), \dots, (x_N, y_N)\\) với các \\(x_i\\) kháu nhau đôi một, luôn tìm được một đa thức \\(P(.)\\) bậc không vượt quá \\(N-1\\) sao cho \\(P(x_i) = y_i, ~\forall i = 1, 2, \dots, N\\). Chẳng phải điều này giống với việc ta đi tìm một mô hình phù hợp (fit) với dữ liệu trong bài toán [Supervised Learning](/2016/12/27/categories/#supervised-learning-hoc-co-giam-sat) hay sao? Thậm chí điều này còn tốt hơn vì trong Supervised Learning ta chỉ cần xấp xỉ thôi.

Sự thật là nếu một mô hình _quá fit_ với dữ liệu thì nó sẽ gây phản tác dụng! Hiện tượng _quá fit_ này trong Machine Learning được gọi là _overfitting_, là điều mà khi xây dựng mô hình, chúng ta luôn cần tránh. Để có cái nhìn đầu tiên về overfitting, chúng ta cùng xem Hình dưới đây. Có 50 điểm dữ liệu được tạo bằng một đa thức bậc ba cộng thêm nhiễu. Tập dữ liệu này được chia làm hai, 30 điểm dữ liệu màu đỏ cho training data, 20 điểm dữ liệu màu vàng cho test data. Đồ thị của đa thức bậc ba này được cho bởi đường màu xanh lục. Bài toán của chúng ta là giả sử ta không biết mô hình ban đầu mà chỉ biết các điểm dữ liệu, hãy tìm một mô hình "tốt" để mô tả dữ liệu đã cho.

<hr>
<div>
<table width = "100%" style = "border: 0px solid white">
   <tr >
        <td width="40%" style = "border: 0px solid white">
        <img style="display:block;" width = "100%" src = "/assets/15_overfitting/linreg_2.png">
         </td>
        <td width="40%" style = "border: 0px solid white">
        <img style="display:block;" width = "100%" src = "/assets/15_overfitting/linreg_4.png">
        </td>

    </tr>

    <tr >
        <td width="40%" style = "border: 0px solid white">
        <img style="display:block;" width = "100%" src = "/assets/15_overfitting/linreg_8.png">
         </td>
        <td width="40%" style = "border: 0px solid white">
        <img style="display:block;" width = "100%" src = "/assets/15_overfitting/linreg_16.png">
        </td>

    </tr>
</table>
<div class = "thecap"> Underfitting và Overfitting với Polynomial Regression (<a href = "https://github.com/tiepvupsu/tiepvupsu.github.io/blob/master/assets/15_overfitting/LinReg.ipynb">Source code</a>).
</div>
</div>
<hr>

Với những gì chúng ta đã biết từ bài [Linear Regression](/2016/12/28/linearregression/#cac-bai-toan-co-the-giai-bang-linear-regression), với loại dữ liệu này, chúng ta có thể áp dụng [Polynomial Regression](https://en.wikipedia.org/wiki/Polynomial_regression). Bài toán này hoàn toàn có thể được giải quyết bằng Linear Regression với dữ liệu mở rộng cho một cặp điểm \\((x, y)\\) là \\((\mathbf{x}, y)\\) với \\(\mathbf{x} = [1, x, x^2, x^3, \dots, x^d]^T\\) cho đa thức bậc \\(d\\). Điều quan trọng là chúng ta cần tìm bậc \\(d\\) của đa thức cần tìm.

Rõ ràng là một đa thức bậc không vượt quá 29 có thể _fit_ được hoàn toàn với 30 điểm trong training data. Chúng ta cùng xét vài giá trị \\(d = 2, 4, 8, 16\\). Với \\(d = 2\\), mô hình không thực sự tốt vì mô hình dự đoán quá khác so với mô hình thực. Trong trường hợp này, ta nói mô hình bị _underfitting_. Với \\(d = 8\\), với các điểm dữ liệu trong khoảng của training data, mô hình dự đoán và mô hình thực là khá giống nhau. Tuy nhiên, về phía phải, đa thức bậc 8 cho kết quả hoàn toàn ngược với _xu hướng của dữ liệu_. Điều tương tự xảy ra trong trường hợp \\(d = 16\\). Đa thức bậc 16 này _quá fit_ dữ liệu trong khoảng đang xét, và _quá fit_, tức _không được mượt_ trong khoảng dữ liệu training. Việc _quá fit_ trong trường hợp bậc 16 không tốt vì mô hình đang cố gắng mô tả _nhiễu_ hơn là dữ liệu. Hai trường hợp đa thức bậc cao này được gọi là _Overfitting_.

_Nếu bạn nào biết về Đa thức nội suy Lagrange thì có thể hiểu được hiện tượng sai số lớn với các điểm nằm ngoài khoảng của các điểm đã cho. Đó chính là lý do phương pháp đó có từ "nội suy", với các trường hợp "ngoại suy", kết quả thường không chính xác._

Với \\(d = 4\\), ta được mô hình dự đoán khá giống với mô hình thực. Hệ số bậc cao nhất tìm được rất gần với 0 (xem kết quả trong [source code](https://github.com/tiepvupsu/tiepvupsu.github.io/blob/master/assets/15_overfitting/LinReg.ipynb)), vì vậy đa thưc bậc 4 này khá gần với đa thức bậc 3 ban đầu. Đây chính là một mô hình tốt.

Overfitting là hiện tượng mô hình tìm được _quá khớp_ với dữ liệu training. Việc _quá khớp_ này có thể dẫn đến việc dự đoán nhầm nhiễu, và chất lượng mô hình không còn tốt trên dữ liệu test nữa. [Dữ liệu test được giả sử là không được biết trước, và không được sử dụng để xây dựng các mô hình Machine Learning](/general/2017/02/06/featureengineering/#main-algorithms).

Về cơ bản, overfitting xảy ra khi mô hình quá phức tạp để mô phỏng training data. Điều này đặc biệt xảy ra khi lượng dữ liệu training quá nhỏ trong khi độ phức tạp của mô hình quá cao. Trong ví dụ trên đây, độ phức tạp của mô hình có thể được coi là bậc của đa thức cần tìm. Trong [Multi-layer Perceptron](/2017/02/24/mlp/), độ phức tạp của mô hình có thể được coi là số lượng hidden layers và số lượng units trong các hidden layers.

Vậy, có những kỹ thuật nào giúp tránh Overfitting?

 Trước hết, chúng ta cần một vài đại lượng để đánh giá chất lượng của mô hình trên training data và test data. Dưới đây là hai đại lượng đơn giản, với giả sử \\(\mathbf{y}\\) là đầu ra thực sự (có thể là vector), và \\(\mathbf{\hat{y}}\\) là đầu ra dự đoán bởi mô hình:

**Train error:** Thường là hàm mất mát áp dụng lên training data. Hàm mất mát này cần có một thừa số \\(\frac{1}{N\_{\text{train}}} \\) để tính giá trị trung bình, tức mất mát trung bình trên mỗi điểm dữ liệu. Với Regression, đại lượng này thường được định nghĩa:
\\[
\text{train error}= \frac{1}{N\_{\text{train}}} \sum_{\text{training set}} \\|\mathbf{y} - \mathbf{\hat{y}}\\|_p^2
\\]
với \\(p\\) [thường bằng 1 hoặc 2](/math/#mot-so-chuan-thuong-dung).

Với Classification, trung bình cộng của [cross entropy](/2017/02/17/softmax/#-cross-entropy) có thể được sử dụng.

**Test error:** Tương tự như trên nhưng áp dụng mô hình tìm được vào **test data**. Chú ý rằng, khi xây dựng mô hình, ta không được sử dụng thông tin trong tập dữ liệu test. Dữ liệu test chỉ được dùng để đánh giá mô hình. Với Regression, đại lượng này thường được định nghĩa:
\\[
\text{test error}= \frac{1}{N\_{\text{test}}} \sum_{\text{test set}} \\|\mathbf{y} - \mathbf{\hat{y}}\\|_p^2
\\]

với \\(p\\) giống như \\(p\\) trong cách tính _train error_ phía trên.

_Việc lấy trung bình là quan trọng vì lượng dữ liệu trong hai tập hợp training và test có thể chênh lệch rất nhiều._

Một mô hình được coi là tốt (fit) nếu cả _train error_ và _test error_ đều thấp. Nếu _train error_ thấp nhưng _test error_ cao, ta nói mô hình bị overfitting. Nếu _train error_ cao và _test error_ cao, ta nói mô hình bị underfitting. Nếu _train error_ cao nhưng _test error_ thấp, tôi không biết tên của mô hình này, vì cực kỳ may mắn thì hiện tượng này mới xảy ra, hoặc có chỉ khi tập dữ liệu test quá nhỏ.

Chúng ta cùng đi vào phương pháp đầu tiên

<a name="-validation"></a>

<a name="-validation"></a>
## 2. Validation
<a name="-validation-1"></a>

<a name="-validation-1"></a>
### 2.1. Validation
Chúng ta vẫn quen với việc chia tập dữ liệu ra thành hai tập nhỏ: training data và test data. Và một điều tôi vẫn muốn nhắc lại là khi xây dựng mô hình, ta không được sử dụng test data. Vậy làm cách nào để biết được chất lượng của mô hình với _unseen data_ (tức dữ liệu chưa nhìn thấy bao giờ)?

Phương pháp đơn giản nhất là _trích_ từ tập training data ra một tập con nhỏ và thực hiện việc đánh giá mô hình trên tập con nhỏ này. Tập con nhỏ __được trích ra từ training set__ này được gọi là _validation set_. Lúc này, __training set là phần còn lại của training set ban đầu__. Train error được tính trên training set mới này, và có một khái niệm nữa được định nghĩa tương tự như trên _validation error_, tức error được tính trên tập validation.

> Việc này giống như khi bạn ôn thi. Giả sử bạn không biết đề thi như thế nào nhưng có 10 bộ đề thi từ các năm trước. Để xem trình độ của mình trước khi thi thế nào, có một cách là bỏ riêng một bộ đề ra, không ôn tập gì. Việc ôn tập sẽ được thực hiện dựa trên 9 bộ còn lại. Sau khi ôn tập xong, bạn bỏ bộ đề đã để riêng ra làm thử và kiểm tra kết quả, như thế mới "khách quan", mới giống như thi thật. 10 bộ đề ở các năm trước là "toàn bộ" training set bạn có. Để tránh việc học lệch, học tủ theo chỉ 10 bộ, bạn tách 9 bộ ra làm training set thật, bộ còn lại là validation test. Khi làm như thế thì mới đánh giá được việc bạn học đã tốt thật hay chưa, hay chỉ là _học tủ_. Vì vậy, _Overfitting_ còn có thể so sánh với việc _Học tủ_ của con người. 



Với khái niệm mới này, ta tìm mô hình sao cho cả _train eror_ và _validation error_ đều nhỏ, qua đó có thể dự đoán được rằng _test error_ cũng nhỏ. Phương pháp thường được sử dụng là sử dụng nhiều mô hình khác nhau. Mô hình nào cho _validation error_ nhỏ nhất sẽ là mô hình tốt.

Thông thường, ta bắt đầu từ mô hình đơn giản, sau đó tăng dần độ phức tạp của mô hình. Tới khi nào _validation error_ có chiều hướng tăng lên thì chọn mô hình ngay trước đó. Chú ý rằng mô hình càng phức tạp, _train error_ có xu hướng càng nhỏ đi.

Hính dưới đây mô tả ví dụ phía trên với bậc của đa thức tăng từ 1 đến 8. Tập validation bao gồm 10 điểm được lấy ra từ tập training ban đầu.

<div class="imgcap">
<img src ="\assets\15_overfitting\linreg_val.png" align = "center" width = "500">
<div class = "thecap">Hình 2: Lựa chọn mô hình dựa trên validation (<a href = "https://github.com/tiepvupsu/tiepvupsu.github.io/blob/master/assets/15_overfitting/LinReg-validation.ipynb">Source code</a>).</div>
</div>
Chúng ta hãy tạm chỉ xét hai đường màu lam và đỏ, tương ứng với _train error_ và _validation error_. Khi bậc của đa thức tăng lên, _train error_ có xu hướng giảm. Điều này dễ hiểu vì đa thức bậc càng cao, dữ liệu càng được _fit_. Quan sát đường màu đỏ, khi bậc của đa thức là 3 hoặc 4 thì _validation error_ thấp, sau đó tăng dần lên. Dựa vào _validation error_, ta có thể xác định được bậc cần chọn là 3 hoặc 4. Quan sát tiếp đường màu lục, tương ứng với _test error_, thật là trùng hợp, với bậc bằng 3 hoặc 4, _test error_ cũng đạt giá trị nhỏ nhất, sau đó tăng dần lên. Vậy cách làm này ở đây đã tỏ ra hiệu quả.

Việc không sử dụng _test data_ khi lựa chọn mô hình ở trên nhưng vẫn có được kết quả khả quan vì ta giả sử rằng _validation data_ và _test data_ có chung một đặc điểm nào đó. Và khi cả hai đều là _unseen data_, _error_ trên hai tập này sẽ tương đối giống nhau.

Nhắc lại rằng, khi bậc nhỏ (bằng 1 hoặc 2), cả ba error đều cao, ta nói mô hình bị _underfitting_.


<a name="-cross-validation"></a>

<a name="-cross-validation"></a>
### 2.2. Cross-validation
Trong nhiều trường hợp, chúng ta có rất hạn chế số lượng dữ liệu để xây dựng mô hình. Nếu lấy quá nhiều dữ liệu trong tập training ra làm dữ liệu validation, phần dữ liệu còn lại của tập training là không đủ để xây dựng mô hình. Lúc này, tập validation phải thật nhỏ để giữ được lượng dữ liệu cho training đủ lớn. Tuy nhiên, một vấn đề khác nảy sinh. Khi tập validation quá nhỏ, hiện tượng overfitting lại có thể xảy ra với tập training còn lại. Có giải pháp nào cho tình huống này không?

Câu trả lời là _cross-validation_.

_Cross validation_ là một cải tiến của _validation_ với lượng dữ liệu trong tập validation là nhỏ nhưng chất lượng mô hình được đánh giá trên nhiều tập _validation_ khác nhau. Một cách thường đường sử dụng là chia tập training ra \\(k\\) tập con không có phần tử chung, có kích thước gần bằng nhau. Tại mỗi lần kiểm thử , được gọi là _run_, một trong số \\(k\\) tập con được lấy ra làm _validata set_. Mô hình sẽ được xây dựng dựa vào hợp của \\(k-1\\) tập con còn lại. Mô hình cuối được xác định dựa trên trung bình của các _train error_ và _validation error_. Cách làm này còn có tên gọi là __k-fold cross validation__.

Khi \\(k\\) bằng với số lượng phần tử trong tập _training_ ban đầu, tức mỗi tập con có đúng 1 phần tử, ta gọi kỹ thuật này là __leave-one-out__.

Sklearn hỗ trợ rất nhiều phương thức cho phân chia dữ liệu và tính toán _scores_ của các mô hình. Bạn đọc có thể xem thêm tại [Cross-validation: evaluating estimator performance](http://scikit-learn.org/stable/modules/cross_validation.html).




<a name="-regularization"></a>

<a name="-regularization"></a>
## 3. Regularization


Một nhược điểm lớn của _cross-validation_ là số lượng _training runs_ tỉ lệ thuận với \\(k\\). Điều đáng nói là mô hình polynomial như trên chỉ có một tham số cần xác định là bậc của đa thức. Trong các bài toán Machine Learning, lượng tham số cần xác định thường lớn hơn nhiều, và khoảng giá trị của mỗi tham số cũng rộng hơn nhiều, chưa kể đến việc có những tham số có thể là số thực. Như vậy, việc chỉ xây dựng một mô hình thôi cũng là đã rất phức tạp rồi. Có một cách giúp số mô hình cần huấn luyện giảm đi nhiều, thậm chí chỉ một mô hình. Cách này có tên gọi chung là _regularization_.

_Regularization_, một cách cơ bản, là thay đổi mô hình một chút để tránh overfitting trong khi vẫn giữ được tính tổng quát của nó (tính tổng quát là tính mô tả được nhiều dữ liệu, trong cả tập training và test). Một cách cụ thể hơn, ta sẽ tìm cách _di chuyển_ nghiệm của bài toán tối ưu hàm mất mát tới một điểm gần nó. Hướng di chuyển sẽ là hướng làm cho mô hình _ít phức tạp hơn_ mặc dù giá trị của hàm mất mát có tăng lên một chút.

Một kỹ thuật rất đơn giản là _early stopping_.


<a name="-early-stopping"></a>

<a name="-early-stopping"></a>
### 3.1. Early Stopping
Trong nhiều bài toán Machine Learning, chúng ta cần sử dụng các thuật toán lặp để tìm ra nghiệm, ví dụ như Gradient Descent. Nhìn chung, hàm mất mát giảm dần khi số vòng lặp tăng lên. Early stopping tức dừng thuật toán trước khi hàm mất mát đạt giá trị quá nhỏ, giúp tránh overfitting.

Vậy dừng khi nào là phù hợp?

Một kỹ thuật thường được sử dụng là tách từ training set ra một tập validation set như trên. Sau một (hoặc một số, ví dụ 50) vòng lặp, ta tính cả _train error_ và _validation error_, đến khi _validation error_ có chiều hướng tăng lên thì dừng lại, và quay lại sử dụng mô hình tương ứng với điểm và _validation error_ đạt giá trị nhỏ.

<div class="imgcap">
<img src ="\assets\15_overfitting\EarlyStopping.png" align = "center" width = "400">
<div class = "thecap">Hình 3: Early Stopping. Đường màu xanh là <i>train error</i>, đường màu đỏ là <i>validation error</i>. Trục x là số lượng vòng lặp, trục y là error. Mô hình được xác định tại vòng lặp mà <i>validation error</i> đạt giá trị nhỏ nhất.  (<a href = "https://en.wikipedia.org/wiki/Overfitting">Overfitting - Wikipedia</a>).</div>
</div>
Hình trên đây mô tả cách tìm điểm _stopping_. Chúng ta thấy rằng phương pháp này khá giống với phương pháp tìm bậc của đa thức ở phần trên của bài viết.

<a name="-them-so-hang-vao-ham-mat-mat"></a>

<a name="-them-so-hang-vao-ham-mat-mat"></a>
### 3.2. Thêm số hạng vào hàm mất mát


Kỹ thuật regularization phổ biến nhất là thêm vào hàm mất mát một số hạng nữa. Số hạng này thường dùng để đánh giá độ phức tạp của mô hình. Số hạng này càng lớn, thì mô hình càng phức tạp. _Hàm mất mát mới_ này thường được gọi là __regularized loss function__, thường được định nghĩa như sau:
\\[
J_{\text{reg}}(\theta) = J(\theta) + \lambda R(\theta)
\\]

Nhắc lại rằng \\(\theta\\) được dùng để ký hiệu các biến trong mô hình, chẳng hạn như các hệ số \\(\mathbf{w}\\) trong Neural Networks. \\(J(\theta)\\) ký hiệu cho hàm mất mát (_loss function_) và \\(R(\theta)\\) là số hạng _regularization_. \\(\lambda\\) thường là một số dương để cân bằng giữa hai đại lượng ở vế phải.


Việc tối thiểu _regularized loss function_, nói một cách tương đối, đồng nghĩa với việc tối thiểu cả _loss function_ và số hạng _regularization_. Tôi dùng cụm "nói một cách tương đối" vì nghiệm của bài toán tối ưu _loss function_ và __regularized loss function__ là khác nhau.  Chúng ta vẫn mong muốn rằng sự khác nhau này là nhỏ, vì vậy tham số regularization (_regularizaton parameter_) \\(\lambda\\) thường được chọn là một số nhỏ để biểu thức regularization không làm giảm quá nhiều chất lượng của nghiệm.

Với các mô hình Neural Networks, một số kỹ thuật regularization thường được sử dụng là:

<a name="-\\l\\-regularization"></a>

<a name="-%5C%5Cl%5C%5C-regularization"></a>
### 3.3. \\(l_2\\) regularization
Trong kỹ thuật này:
\\[
R(\mathbf{w}) = \\|\mathbf{w}\\|_2^2
\\]
tức norm 2 của hệ số.

_Nếu bạn đọc chưa quen thuộc với khái niệm norm, bạn được khuyến khích đọc [phần phụ lục này](/math/#-norms-chuan)_.

Hàm số này có một vài đặc điểm đang lưu ý:

* Thứ nhất, \\(\\|\mathbf{w}\\|\_2^2\\) là một hàm số _rất mượt_, tức có đạo hàm tại mọi , đạo hàm của nó đơn giản là \\(\mathbf{w}\\), vì vậy đạo hàm của _regularized loss function_ cũng rất dễ tính, chúng ta có thể hoàn toàn dùng các phương pháp dựa trên gradient để cập nhật nghiệm. Cụ thể:
\\[
\frac{\partial J_{\text{reg}} }{\partial \mathbf{w}} = \frac{\partial J}{\partial \mathbf{w}} + \lambda \mathbf{w}
\\]
* Thứ hai, việc tối thiểu \\(\\|\mathbf{w}\\|\_2^2\\) đồng nghĩa với việc khiến cho các giá trị của hệ số \\(\mathbf{w}\\) trở nên nhỏ gần với 0. Với Polynomial Regression, việc các hệ số này nhỏ có thể giúp các hệ số ứng với các số hạng bậc cao là nhỏ, giúp tránh overfitting. Với Multi-layer Pereceptron, việc các hệ số này nhỏ giúp cho nhiều hệ số trong các ma trận trọng số là nhỏ. Điều này tương ứng với việc số lượng các hidden units _hoạt động_ (khác không) là nhỏ, cũng giúp cho MLP tránh được hiện tượng overfitting.

\\(l_2\\) regularization là kỹ thuật được sử dụng nhiều nhất để giúp Neural Networks tránh được overfitting. Nó còn có tên gọi khác là __weight decay__. _Decay_ có nghĩa là _tiêu biến_.

Trong Xác suất thống kê, Linear Regression với \\(l_2\\) regularization được gọi là [__Ridge Regression__](https://en.wikipedia.org/wiki/Tikhonov_regularization). Hàm mất mát của _Ridge Regression_ có dạng:
\\[
J(\mathbf{w}) = \frac{1}{2} \\|\mathbf{y} - \mathbf{Xw}\\|_2^2 + \lambda \\|\mathbf{w}\\|_2^2
\\]
trong đó, số hạng đầu tiên ở vế phải chính là hàm mất mát của Linear Regression. Số hạng thứ hai chính là phần regularization.

<a name="vi-du-ve-weight-decay-voi-mlp"></a>

<a name="vi-du-ve-weight-decay-voi-mlp"></a>
#### Ví dụ về Weight Decay với MLP
Chúng ta sử dụng [mô hình MLP giống như bài trước](/2017/02/24/mlp/#-vi-du-tren-python) nhưng dữ liệu có khác đi đôi chút.

```python
# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import math
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(4)

means = [[-1, -1], [1, -1], [0, 1]]
cov = [[1, 0], [0, 1]]
N = 20
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)
```
Dữ liệu được tạo là ba cụm tuân theo phân phối chuẩn có tâm ở `[[-1, -1], [1, -1], [0, 1]]`.

Trong ví dụ này, chúng ta sử dụng số hạng regularization:
\\[
\lambda R(\mathbf{W}) = \lambda \sum_{l=1}^L \\|\mathbf{W}^{(l)}\\|_F^2
\\]

với \\(\\|.\\|_F\\) là [Frobenius norm](/math/#cho-ma-tran), là căn bậc hai của tổng bình phường các phẩn tử của ma trận.

(Bạn đọc được khuyến khích đọc bài [MLP](/2017/02/24/mlp/) để hiểu các ký hiệu).

Chú ý rằng weight decay ít khi được áp dụng lên biases. Tôi thay đổi tham số regularization \\(\lambda\\) và nhận được kết quả như sau:

<hr>
<div>
<table width = "100%" style = "border: 0px solid white">
   <tr >
        <td width="40%" style = "border: 0px solid white">
        <img style="display:block;" width = "100%" src = "/assets/15_overfitting/nnet_reg0.png">
         </td>
        <td width="40%" style = "border: 0px solid white">
        <img style="display:block;" width = "100%" src = "/assets/15_overfitting/nnet_reg0.001.png">
        </td>

    </tr>

    <tr >
        <td width="40%" style = "border: 0px solid white">
        <img style="display:block;" width = "100%" src = "/assets/15_overfitting/nnet_reg0.01.png">
         </td>
        <td width="40%" style = "border: 0px solid white">
        <img style="display:block;" width = "100%" src = "/assets/15_overfitting/nnet_reg0.1.png">
        </td>

    </tr>
</table>
<div class = "thecap"> Multi-layer Perceptron với Weight Decay (<a href = "https://github.com/tiepvupsu/tiepvupsu.github.io/blob/master/assets/15_overfitting/Weight%20Decay.ipynb">Source code</a>).
</div>
</div>
<hr>
Khi \\(\lambda = 0\\), tức không có regularization, ta nhận thấy gần như toàn bộ dữ liệu trong tập training được phân lớp đúng. Việc này khiến cho các class bị phân làm nhiều mảnh không được tự nhiên. Khi \\(\lambda = 0.001\\), vẫn là một số nhỏ, các đường phân chia trông tự nhiên hơn, nhưng lớp màu xanh lam vẫn bị chia làm hai bởi lớp màu xanh lục. Đây chính là biểu hiện của overfitting.

Khi \\(\lambda\\) tăng lên, tức sự ảnh hưởng của regularization tăng lên (xem hàng dưới), đường ranh giới giữa các lớp trở lên tự nhiên hơn. Nói cách khác, với \\(\lambda\\) đủ lớn, weight decay có tác dụng hạn chế overfitting trong MLP.

Bạn đọc hãy thử vào trong [Source code](https://github.com/tiepvupsu/tiepvupsu.github.io/blob/master/assets/15_overfitting/Weight%20Decay.ipynb), thay \\(\lambda = 1\\) bằng cách thay dòng cuối cùng:
```python
mynet(1)
```
rồi chạy lại toàn bộ code, xem các đường phân lớp trông như thế nào. Gợi ý: _underfitting_.

Khi \\(\lambda\\) quá lớn, tức ta xem phần _regularization_ quan trọng hơn phần _loss fucntion_, một hiện tượng xấu xảy ra là các phần tử của \\(\mathbf{w}\\) tiến về 0 để thỏa mãn regularization là nhỏ.

[Sklearn có cung cấp rất nhiều chức năng cho MLP](http://scikit-learn.org/stable/modules/neural_networks_supervised.html), trong đó ta có thể lựa chọn số lượng hidden layers và số lượng hidden units trong mỗi layer, activation functions, weight decay, [learning rate, hệ số momentum, nesterovs_momentum](/2017/01/16/gradientdescent2/), có early stopping hay không, lượng dữ liệu được tách ra làm validation set, và nhiều chức năng khác.

<a name="-tikhonov-regularization"></a>

<a name="-tikhonov-regularization"></a>
### 3.4. Tikhonov regularization
\\[
\lambda R(\mathbf{w}) = \\|\Gamma \mathbf{w}\\|_2^2
\\]

Với \\(\Gamma\\) (viết hoa của gamma) là một ma trận. Ma trận \\(\Gamma\\) hay được dùng nhất là ma trận đường chéo. Nhận thấy rằng \\(l_2\\) regularization chính là một trường hợp đặc biệt của Tikhonov regularization với \\(\Gamma = \lambda \mathbf{I}\\) với \\(\mathbf{I}\\) là ma trận đơn vị (_the identity matrix_), tức các phần tử trên đường chéo của \\(\Gamma\\) là như nhau.

Khi các phần tử trên đường chéo của \\(\Gamma\\) là khác nhau, ta có một phiên bản gọi là _weighted \\(l_2\\) regularization_, tức đánh trọng số khác nhau cho mỗi phần tử trong \\(\mathbf{w}\\). Phần tử nào càng bị đánh trọng số cao thì nghiệm tương ứng càng nhỏ (để đảm bảo rằng hàm mất mát là nhỏ). Với Polynomial Regression, các phần tử ứng với hệ số bậc cao sẽ được đánh trọng số cao hơn, khiến cho xác suất để chúng gần 0 là lớn hơn.

<a name="-regularizers-for-sparsity"></a>

<a name="-regularizers-for-sparsity"></a>
### 3.5. Regularizers for sparsity

Trong nhiều trường hợp, ta muốn các hệ số _thực sự_ bằng 0 chứ không phải là _nhỏ gần 0_ như \\(l_2\\) regularization đã làm phía trên. Lúc đó, có một regularization khác được sử dụng, đó là \\(l_0\\) regularization:
\\[
R(\mathbf{W}) = \\|\mathbf{w}\\|_0
\\]

Norm 0 không phải là một norm thực sự mà là giả norm. (Bạn được khuyến khích đọc thêm về [norms (chuẩn)](/math/#-norms-chuan)). Norm 0 của một vector là số các phần tử khác không của vector đó. Khi norm 0 nhỏ, tức rất nhiều phần tử trong vector đó bằng 0, ta nói vector đó là _sparse_.

Việc giải bài toán tổi thiểu norm 0 nhìn chung là khó vì hàm số này không _convex_, không liên tục. Thay vào đó, norm 1 thường được sử dụng:
\\[
R(\mathbf{W}) = \\|\mathbf{w}\\|\_1 = \sum_{i=0}^d \|w\_i\|
\\]

Norm 1 là tổng các trị tuyệt đối của tất cả các phần tử. Người ta đã chứng minh được rằng tối thiểu norm 1 sẽ dẫn tới nghiệm có nhiều phần tử bằng 0. Ngoài ra, vì norm 1 là một _norm thực sự_ (proper norm) nên hàm số này là _convex_, và hiển nhiên là liên tục, việc giải bài toán này dễ hơn việc giải bài toán tổi thiểu norm 0. Về \\(l_1\\) regularization, bạn đọc có thể đọc thêm trong [lecture note](\\(l_1\\) regularization) này. Việc giải bài toán \\(l_1\\) regularization nằm ngoài mục đích của tôi trong bài viết này. Tôi hứa sẽ quay lại phần này sau. (Vì đây là phần chính trong nghiên cứu của tôi).

Trong Thống Kê, việc sử dụng \\(l_1\\) regularization còn được gọi là [LASSO](https://en.wikipedia.org/wiki/Lasso_(statistics)) (Least Absolute Shrinkage and Selection Operator)).

Khi cả \\(l_2\\) và \\(l_1\\) regularization được sử dụng, ta có mô hình gọi là [Elastic Net Regression](https://en.wikipedia.org/wiki/Elastic_net_regularization).

<a name="-regularization-trong-sklearn"></a>

<a name="-regularization-trong-sklearn"></a>
### 3.6. Regularization trong sklearn

Trong [sklearn](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), ví dụ [Logistic Regression](/2017/01/27/logisticregression/), bạn cũng có thể sử dụng các \\(l_1\\) và \\(l_2\\) regularizations bằng cách khai báo biến `penalty='l1'` hoặc `penalty = 'l2'` và biến `C`, trong đó `C` là _nghịch đảo_ của \\(\lambda\\). Trong các bài trước khi chưa nói về  Overfitting và Regularization, tôi có sử dụng `C = 1e5` để chỉ ra rằng \\(\lambda\\) là một số rất nhỏ.

<a name="-cac-phuong-phap-khac"></a>

<a name="-cac-phuong-phap-khac"></a>
## 4. Các phương pháp khác
Ngoài các phương pháp đã nêu ở trên, với mỗi mô hình, nhiều phương pháp tránh overfitting khác cũng được sử dụng. Điển hình là [Dropout trong Deep Neural Networks mới được đề xuất gần đây](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf). Một cách ngắn gọn, dropout là một phương pháp _tắt_ ngẫu nhiên các units trong Networks. _Tắt_ tức cho các unit giá trị bằng không và tính toán feedforward và backpropagation bình thường trong khi training. Việc này không những giúp lượng tính toán giảm đi mà còn làm giảm việc overffitng. Tôi xin được quay lại vấn đề này nếu có dịp nói  sâu về Deep Learning trong tương lai.

Bạn đọc có thể tìm đọc thêm với các từ khóa: [pruning](https://en.wikipedia.org/wiki/Pruning_(decision_trees)) (tránh overftting trong Decision Trees), [VC dimension](https://en.wikipedia.org/wiki/VC_dimension) (đo độ phức tạp của mô hình, độ phức tạp càng lớn thì càng dễ bị overfitting).

<a name="-tom-tat-noi-dung"></a>

<a name="-tom-tat-noi-dung"></a>
## 5. Tóm tắt nội dung
* Một mô hình mô tốt là mộ mô hình có _tính tổng quát_, tức mô tả được dữ liệu cả trong lẫn ngoài tập training. Mô hình chỉ mô tả tốt dữ liệu trong tập training được gọi là **overfitting**.

* Để tránh overfitting, có rất nhiều kỹ thuật được sử dụng, điển hình là **cross-validation** và **regularization**. Trong Neural Networks, **weight decay** và **dropout** thường được dùng.

<a name="-tai-lieu-tham-khao"></a>

<a name="-tai-lieu-tham-khao"></a>
## 6. Tài liệu tham khảo

[1] [Overfitting - Wikipedia](https://en.wikipedia.org/wiki/Overfitting)

[2] [Cross-validation - Wikipedia](https://en.wikipedia.org/wiki/Cross-validation_(statistics))

[3] [Pattern Recognition and Machine Learning](users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop - Pattern Recognition And Machine Learning - Springer  2006.pdf)

[4] Krogh, Anders, and John A. Hertz. ["A simple weight decay can improve generalization."](https://papers.nips.cc/paper/563-a-simple-weight-decay-can-improve-generalization.pdf) NIPS. Vol. 4. 1991.

[5] Srivastava, Nitish, et al. ["Dropout: A Simple Way to Prevent Neural Networks from  Overfitting"](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) Journal of Machine Learning Research 15.1 (2014): 1929-1958.
