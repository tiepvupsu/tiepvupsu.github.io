---
layout: post
comments: true
title:  "Bài 25: Matrix Factorization Collaborative Filtering"
title2:  "25. Matrix Factorization Collaborative Filtering"
date:   2017-05-31 15:22:00
permalink: 2017/05/31/matrixfactorization/
mathjax: true
tags: Recommendation-systems, dimensionality-reduction
category: Recommendation-systems
sc_project: 11358048
sc_security: 5bef7cd2
img: /assets/25_mf/mf1.png
summary: Trong bài viết này, chúng ta sẽ làm quen với một hướng tiếp cận khác cho Collaborative Filtering dựa trên Matrix Factorization (hoặc Matrix Decomposition), tức Phân tích ma trận thành nhân tử.
---

**Trong trang này:**
<!-- MarkdownTOC -->

- [1. Giới thiệu](#-gioi-thieu)
- [2. Xây dựng và tối ưu hàm mất mát](#-xay-dung-va-toi-uu-ham-mat-mat)
    - [2.1. Hàm mất mát](#-ham-mat-mat)
    - [2.2. Tối ưu hàm mất mát](#-toi-uu-ham-mat-mat)
- [3. Lập trình Python](#-lap-trinh-python)
    - [3.1. `class MF`](#-class-mf)
    - [3.2. Áp dụng lên MovieLens 100k](#-ap-dung-len-movielens-k)
    - [3.3. Áp dụng lên MovieLens 1M](#-ap-dung-len-movielens-m)
- [4. Thảo luận](#-thao-luan)
    - [4.1. Khi có bias](#-khi-co-bias)
    - [4.2. Nonnegative Matrix Factorization](#-nonnegative-matrix-factorization)
    - [4.3. Incremental Matrix Factorization](#-incremental-matrix-factorization)
    - [4.4. Others](#-others)
- [6. Tài liệu tham khảo](#-tai-lieu-tham-khao)

<!-- /MarkdownTOC -->


<a name="-gioi-thieu"></a>

## 1. Giới thiệu 
Trong [Bài 24](/2017/05/24/collaborativefiltering/), chúng ta đã làm quen với một hướng tiếp cận trong Collaborative Filtering dựa trên hành vi của các _users_ hoặc _items_ lân cận có tên là Neighborhood-based Collaborative Filtering. Trong bài viết này, chúng ta sẽ làm quen với một hướng tiếp cận khác cho Collaborative Filtering dựa trên _Matrix Factorization_ (hoặc _Matrix Decomposition_), tức _Phân tích ma trận thành nhân tử_. 

Nhắc lại rằng trong [Content-based Recommendation Systems](/2017/05/17/contentbasedrecommendersys/), mỗi _item_ được mô tả bằng một vector \\(\mathbf{x}\\) được gọi là _item profile_. Trong phương pháp này, ta cần tìm một vector hệ số \\(\mathbf{w}\\) tương ứng với mỗi _user_ sao cho _rating_ đã biết mà _user_ đó cho _item_ xấp xỉ với: 
\\[
y \approx \mathbf{xw}
\\]

Với cách làm trên, [_Utility Matrix_](/2017/05/17/contentbasedrecommendersys/#-utility-matrix) \\(\mathbf{Y}\\), giả sử đã được điền hết, sẽ xấp xỉ với: 

\\[
\mathbf{Y} \approx \left[ \begin{matrix}
\mathbf{x}_1\mathbf{w}_1 & \mathbf{x}_1\mathbf{w}_2 & \dots & \mathbf{x}\_1 \mathbf{w}_N\\\
\mathbf{x}_2\mathbf{w}_1 & \mathbf{x}_2\mathbf{w}_2 & \dots & \mathbf{x}\_2 \mathbf{w}_N\\\
\dots & \dots & \ddots & \dots \\\
\mathbf{x}_M\mathbf{w}_1 & \mathbf{x}_M\mathbf{w}_2 & \dots & \mathbf{x}\_M \mathbf{w}_N\\\
\end{matrix} \right]
 = \left[ \begin{matrix}
\mathbf{x}_1 \\\
\mathbf{x}_2 \\\
\dots \\\
\mathbf{x}_M \\\
\end{matrix} \right]
\left[ \begin{matrix}
\mathbf{w}_1 & \mathbf{w}_2 & \dots & \mathbf{w}\_N
\end{matrix} \right] = \mathbf{XW}
\\]

với \\(M, N\\) lần lượt l
à số _items_ và số _users_. 

Chú ý rằng, \\(\mathbf{x}\\) được xây dựng dựa trên thông tin mô tả của _item_ và quá trình xây dựng này độc lập với quá trịnh đi tìm hệ số phù hợp cho mỗi _user_. Như vậy, việc xây dựng _item profile_ đóng vai trò rất quan trọng và có ảnh hưởng trực tiếp lên hiệu năng của mô hình. Thêm nữa, việc xây dựng từng mô hình riêng lẻ cho mỗi _user_ dẫn đến kết quả chưa thực sự tốt vì không khai thác được đặc điểm của những _users_ gần giống nhau. 


Bây giờ, giả sử rằng ta không cần xây dựng từ trước các _item profile_ \\(\mathbf{x}\\) mà vector đặc trưng cho mỗi _item_ này có thể được huấn luyện đồng thời với mô hình của mỗi _user_ (ở đây là 1 vector hệ số). Điều này nghĩa là, biến số trong bài toán tối ưu là cả \\(\mathbf{X}\\) và \\(\mathbf{W}\\); trong đó, \\(\mathbf{X}\\) là ma trận của toàn bộ _item profiles_, mỗi **hàng** tương ứng với 1 _item_, \\(\mathbf{W}\\) là ma trận của toàn bộ _user models_, mỗi **cột** tương ứng với 1 _user_. 

Với cách làm này, chúng ta đang cố gắng xấp xỉ _Utility Matrix_ \\(\mathbf{Y} \in \mathbb{R}^{M \times N}\\) bằng tích của hai ma trận \\(\mathbf{X}\in \mathbb{R}^{M\times K}\\) và \\(\mathbf{W} \in \mathbb{R}^{K \times N}\\). 

Thông thường, \\(K\\) được chọn là một số nhỏ hơn rất nhiều so với \\(M, N\\). Khi đó, cả hai ma trận \\(\mathbf{X}\\) và \\(\mathbf{W}\\) đều có rank không vượt quá \\(K\\). Chính vì vậy, phương pháp này còn được gọi là _Low-Rank Matrix Factorization_ (xem Hình 1).
<hr>
<div class="imgcap">
<img src ="/assets/25_mf/mf1.png" align = "center" width = "800">
<div class = "thecap" align = "left">Hình 1: Matrix Factorization. Utility matrix \(\mathbf{Y}\) được phân tích thành tích của hai ma trận low-rank \(\mathbf{X}\) và \\(\mathbf{W}\) </div>
</div> 
<hr>

Có một vài điểm lưu ý ở đây: 

* Ý tưởng chính đằng sau Matrix Factorization cho Recommendation Systems là tồn tại các _latent features_ (tính chất ẩn) mô tả sự liên quan giữa các _items_ và _users_. Ví dụ với hệ thống gợi ý các bộ phim, tính chất ẩn có thể là _hình sự_, _chính trị_, _hành động_, _hài_, ...; cũng có thể là một sự kết hợp nào đó của các thể loại này; hoặc cũng có thể là bất cứ điều gì mà chúng ta không thực sự cần đặt tên. Mỗi _item_ sẽ mang tính chất ẩn ở một mức độ nào đó tương ứng với các hệ số trong vector \\(\mathbf{x}\\) của nó, hệ số càng cao tương ứng với việc mang tính chất đó càng cao. Tương tự, mỗi _user_ cũng sẽ có xu hướng thích những tính chất ẩn nào đó và được mô tả bởi các hệ số trong vector \\(\mathbf{w}\\) của nó. Hệ số cao tương ứng với việc _user_ thích các bộ phim có tính chất ẩn đó. Giá trị của biểu thức \\(\mathbf{xw}\\) sẽ cao nếu các thành phần tương ứng của \\(\mathbf{x}\\) và \\(\mathbf{w}\\) đều cao. Điều này nghĩa là _item_ mang các tính chất ẩn mà _user_ thích, vậy thì nên gợi ý _item_ này cho _user_ đó. 

* Vậy tại sao Matrix Factorization lại được xếp vào Collaborative Filtering? Câu trả lời đến từ việc đi tối ưu hàm mất mát mà chúng ta sẽ thảo luận ở Mục 2. Về cơ bản, để tìm nghiệm của bài toán tối ưu, ta phải lần lượt đi tìm \\(\mathbf{X}\\) và \\(\mathbf{W}\\) khi thành phần còn lại được cố định. Như vậy, mỗi hàng của \\(\mathbf{X}\\) sẽ phụ thuộc vào toàn bộ các cột của \\(\mathbf{W}\\). Ngược lại, mỗi cột của \\(\mathbf{W}\\) lại phục thuộc vào toàn bộ các hàng của \\(\mathbf{X}\\). Như vậy, có những mỗi quan hệ ràng buộc _chằng chịt_ giữa các thành phần của hai ma trận trên. Tức chúng ta cần sử dụng thông tin của tất cả để suy ra tất cả. Vậy nên phương pháp này cũng được xếp vào Collaborative Filtering. 

* Trong các bài toán thực tế, số lượng _items_ \\(M\\) và số lượng _users_ \\(N\\) thường rất lớn. Việc tìm ra các mô hình đơn giản giúp dự đoán _ratings_ cần được thực hiện một cách nhanh nhất có thể. [Neighborhood-based Collaborative Filtering](/2017/05/24/collaborativefiltering/) không yêu cầu việc _learning_ quá nhiều, nhưng trong quá trình dự đoán (_inference_), ta cần đi tìm độ _similarity_ của _user_ đang xét với *toàn bộ* các _users_ còn lại rồi suy ra kết quả. Ngược lại, với Matrix Factorization, việc _learning_ có thể hơi phức tạp một chút vì phải lặp đi lặp lại việc tối ưu một ma trận khi cố định ma trận còn lại, nhưng việc _inference_ đơn giản hơn vì ta chỉ cần lấy tích của hai vector \\(\mathbf{xw}\\), mỗi vector có độ dài \\(K\\) là một số nhỏ hơn nhiều so với \\(M, N\\). Vậy nên quá trình _inference_ không yêu cầu khả năng tính toán cao. Việc này khiến nó phù hợp với các mô hình có tập dữ liệu lớn. 

* Thêm nữa, việc lưu trữ hai ma trận \\(\mathbf{X}\\) và \\(\mathbf{W}\\) yêu cầu lượng bộ nhớ nhỏ khi so với việc lưu toàn bộ _Similarity matrix_ trong Neighborhood-based Collaborative Filtering. Cụ thể, ta cần bộ nhớ để chứa \\(K(M+N)\\) phần tử thay vì lưu \\(M^2\\) hoặc \\(N^2\\) của _Similarity matrix_. 

Tiếp theo, chúng ta cùng đi xây dựng hàm mất mát và cách tối ưu nó. 

<a name="-xay-dung-va-toi-uu-ham-mat-mat"></a>

## 2. Xây dựng và tối ưu hàm mất mát

<a name="-ham-mat-mat"></a>

### 2.1. Hàm mất mát 
[Tương tự như trong Content-based Recommendation Systems](/2017/05/17/contentbasedrecommendersys/#-xay-dung-ham-mat-mat), việc xây dựng hàm mất mát cũng được dựa trên các thành phần đã được điền của Utility Matrix \\(\mathbf{Y}\\), có khác một chút là không có thành phần bias và biến tối ưu là cả \\(\mathbf{X}\\) và \\(\mathbf{W}\\). Việc thêm bias vào sẽ được thảo luận ở Mục 4. Việc xây dựng hàm mất mát cho Matrix Factorization là tương đối dễ hiểu: 

\\[
\mathcal{L}(\mathbf{X}, \mathbf{W}) = \frac{1}{2s} \sum_{n=1}^N \sum_{m:r_{mn} = 1} (y_{mn} - \mathbf{x}_m\mathbf{w}_n)^2 + \frac{\lambda}{2} (\|\|\mathbf{X}\|\|_F^2 + \|\|\mathbf{W}\|\|_F^2) ~~~~~ (1)
\\]

trong đó \\(r_{mn} = 1\\) nếu _item_ thứ \\(m\\) đã được đánh giá bởi _user_ thứ \\(n\\), \\(\|\|\bullet\|\|\_F^2\\) là [Frobineous norm](/math/#chuan-cua-ma-tran), tức căn bậc hai của tổng bình phương tất cả các phần tử của ma trận (giống với norm 2 trong vector), \\(s\\) là toàn bộ số _ratings_ đã có. Thành phần thứ nhất chính là trung bình sai số của mô hình. Thành phần thứ hai trong hàm mất mát phía trên là [\\(l_2\\) regularization](/2017/03/04/overfitting/#-\\l\\-regularization), giúp tránh [overfitting](/2017/03/04/overfitting/). 

>**Lưu ý:** Giá trị _ratings_ thường là các giá trị đã được chuẩn hoá, bằng cách trừ mỗi hàng của Utility Matrix đi trung bình cộng của các giá trị đã biết của hàng đó (item-based) hoặc trừ mỗi cột đi trung bình cộng của các giá trị đã biết trong cột đó (user_based). Trong một số trường hợp nhất định, ta không cần chuẩn hoá ma trận này, nhưng kèm theo đó phải có thêm các kỹ thuật khác để giải quyết vấn đề _thiên lệch_ trong khi _rating_.

Việc tối ưu đồng thời \\(\mathbf{X}, \mathbf{W}\\) là tương đối phức tạp, thay vào đó, phương pháp được sử dụng là lần lượt tối ưu một ma trận trong khi cố định ma trận kia, tới khi hội tụ. 


<a name="-toi-uu-ham-mat-mat"></a>

### 2.2. Tối ưu hàm mất mát 
Khi cố định \\(\mathbf{X}\\), việc tối ưu \\(\mathbf{W}\\) chính là bài toán tối ưu trong Content-based Recommendation Systems: 

\\[
\mathcal{L}(\mathbf{W}) = \frac{1}{2s} \sum_{n=1}^N \sum_{m:r_{mn} = 1} (y_{mn} - \mathbf{x}_m\mathbf{w}_n)^2 + \frac{\lambda}{2} \|\|\mathbf{W}\|\|_F^2 ~~~~~ (2)
\\]

Khi cố định \\(\mathbf{W}\\), việc tối ưu \\(\mathbf{X}\\) được đưa về tối ưu hàm: 

\\[
\mathcal{L}(\mathbf{X}) = \frac{1}{2s} \sum_{n=1}^N \sum_{m:r_{mn} = 1} (y_{mn} - \mathbf{x}_m\mathbf{w}_n)^2 + \frac{\lambda}{2} \|\|\mathbf{X}\|\|_F^2 ~~~~~ (3)
\\]

Hai bài toán này sẽ được tối ưu bằng [Gradient Descent](/2017/01/12/gradientdescent/).

Chúng ta có thể thấy rằng bài toán \\((2)\\) có thể được tách thành \\(N\\) bài toán nhỏ, mỗi bài toán ứng với việc đi tối ưu một cột của ma trận \\(\mathbf{W}\\): 
\\[
\mathcal{L}(\mathbf{w}\_n) = \frac{1}{2s} \sum_{m:r_{mn} = 1} (y_{mn} - \mathbf{x}_m\mathbf{w}_n)^2 + \frac{\lambda}{2}\|\|\mathbf{w}_n\|\|_2^2 ~~~~ (4)
\\]

Vì biểu thức trong dấu \\(\sum\\) chỉ phụ thuộc vào các _items_ đã được _rated_ bởi _user_ đang xét, ta có thể đơn giản nó bằng cách đặt \\(\hat{\mathbf{X}}\_n\\) là ma trận được tạo bởi các hàng tương ứng với các _items_ đã được _rated_ đó, và \\(\hat{\mathbf{y}}\_n\\) là các _ratings_ tương ứng. Khi đó: 
\\[
\mathcal{L}(\mathbf{w}\_n) = \frac{1}{2s} \|\|\hat{\mathbf{y}}_n - \hat{\mathbf{X}}_n\mathbf{w}_n\|\|^2 + \frac{\lambda}{2}\|\|\mathbf{w}_n\|\|_2^2 ~~~~~(5)
\\]
và đạo hàm của nó: 
\\[
\frac{\partial \mathcal{L}(\mathbf{w}\_n)}{\partial \mathbf{w}_n} = -\frac{1}{s}\hat{\mathbf{X}}_n^T(\hat{\mathbf{y}}_n - \hat{\mathbf{X}}_n\mathbf{w}_n) + \lambda \mathbf{w}_n ~~~~~ (6)
\\]

**Vậy công thức cập nhật cho mỗi cột của \\(\mathbf{W}\\) là:** 
\\[
\mathbf{w}_n = \mathbf{w}_n - \eta \left(-\frac{1}{s}\hat{\mathbf{X}}_n^T (\hat{\mathbf{y}}_n - \hat{\mathbf{X}}_n\mathbf{w}_n) + \lambda \mathbf{w}_n\right) ~~~~~(7)
\\]

Tương tự như thế, mỗi cột của \\(\mathbf{X}\\), tức vector cho mỗi _item_, sẽ được tìm bằng cách tối ưu: 
\\[
\begin{eqnarray}
\mathcal{L}(\mathbf{x}\_m) &=& \frac{1}{2s}\sum_{n:r_{mn} = 1} (y_{mn} - \mathbf{x}_m\mathbf{w}_n)^2 + \frac{\lambda}{2}\|\|\mathbf{x}_m\|\|_2^2 ~~~~ (8)
\end{eqnarray}
\\]

Đặt \\(\hat{\mathbf{W}}\_m\\) là ma trận được tạo bằng các cột của \\(\mathbf{W}\\) ứng với các _users_ đã đánh giá _item_ đó và \\(\hat{\mathbf{y}}^m\\) là vector _ratings_ tương ứng. \\((8)\\) trở thành: 

\\[
\mathcal{L}(\mathbf{x}\_m)
 = \frac{1}{2s}\|\|\hat{\mathbf{y}}^m - {\mathbf{x}}_m\hat{\mathbf{W}}_m\|\|_2^2 + \frac{\lambda}{2} \|\|\mathbf{x}_m\|\|_2^2 ~~~~~ (9)
\\]
Tương tự như trên, **công thức cập nhật cho mồi hàng của \\(\mathbf{X}\\) sẽ có dạng:**
\\[
\mathbf{x}_m = \mathbf{x}_m - \eta\left(-\frac{1}{s}(\hat{\mathbf{y}}^m - \mathbf{x}_m\hat{\mathbf{W}}_m)\hat{\mathbf{W}}_m^T + \lambda \mathbf{x}_m\right) ~~~~~ (10)
\\]

_Bạn đọc có thể muốn xem thêm [Đạo hàm của hàm nhiều biến](/math/#-dao-ham-cua-ham-nhieu-bien)_
<a name="-lap-trinh-python"></a>

## 3. Lập trình Python 
Tiếp theo, chúng ta sẽ đi sâu vào phần lập trình.

<a name="-class-mf"></a>

### 3.1. `class MF`

**Khởi tạo và chuẩn hoá:**

```python
import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse 

class MF(object):
    """docstring for CF"""
    def __init__(self, Y_data, K, lam = 0.1, Xinit = None, Winit = None, 
            learning_rate = 0.5, max_iter = 1000, print_every = 100, user_based = 1):
        self.Y_raw_data = Y_data
        self.K = K
        # regularization parameter
        self.lam = lam
        # learning rate for gradient descent
        self.learning_rate = learning_rate
        # maximum number of iterations
        self.max_iter = max_iter
        # print results after print_every iterations
        self.print_every = print_every
        # user-based or item-based
        self.user_based = user_based
        # number of users, items, and ratings. Remember to add 1 since id starts from 0
        self.n_users = int(np.max(Y_data[:, 0])) + 1 
        self.n_items = int(np.max(Y_data[:, 1])) + 1
        self.n_ratings = Y_data.shape[0]
        
        if Xinit is None: # new
            self.X = np.random.randn(self.n_items, K)
        else: # or from saved data
            self.X = Xinit 
        
        if Winit is None: 
            self.W = np.random.randn(K, self.n_users)
        else: # from daved data
            self.W = Winit
            
        # normalized data, update later in normalized_Y function
        self.Y_data_n = self.Y_raw_data.copy()


    def normalize_Y(self):
        if self.user_based:
            user_col = 0
            item_col = 1
            n_objects = self.n_users

        # if we want to normalize based on item, just switch first two columns of data
        else: # item bas
            user_col = 1
            item_col = 0 
            n_objects = self.n_items

        users = self.Y_raw_data[:, user_col] 
        self.mu = np.zeros((n_objects,))
        for n in xrange(n_objects):
            # row indices of rating done by user n
            # since indices need to be integers, we need to convert
            ids = np.where(users == n)[0].astype(np.int32)
            # indices of all ratings associated with user n
            item_ids = self.Y_data_n[ids, item_col] 
            # and the corresponding ratings 
            ratings = self.Y_data_n[ids, 2]
            # take mean
            m = np.mean(ratings) 
            if np.isnan(m):
                m = 0 # to avoid empty array and nan value
            self.mu[n] = m
            # normalize
            self.Y_data_n[ids, 2] = ratings - self.mu[n]
```

**Tính giá trị hàm mất mát:**

```python            
    def loss(self):
        L = 0 
        for i in xrange(self.n_ratings):
            # user, item, rating
            n, m, rate = int(self.Y_data_n[i, 0]), int(self.Y_data_n[i, 1]), self.Y_data_n[i, 2]
            L += 0.5*(rate - self.X[m, :].dot(self.W[:, n]))**2
        
        # take average
        L /= self.n_ratings
        # regularization, don't ever forget this 
        L += 0.5*self.lam*(np.linalg.norm(self.X, 'fro') + np.linalg.norm(self.W, 'fro'))
        return L 
```

**Xác định các _items_ được đánh giá bởi 1 _user_, và _users_ đã đánh giá 1 _item_ và các _ratings_ tương ứng:**   

```python    
    def get_items_rated_by_user(self, user_id):
        """
        get all items which are rated by user user_id, and the corresponding ratings
        """
        ids = np.where(self.Y_data_n[:,0] == user_id)[0] 
        item_ids = self.Y_data_n[ids, 1].astype(np.int32) # indices need to be integers
        ratings = self.Y_data_n[ids, 2]
        return (item_ids, ratings)
        
        
    def get_users_who_rate_item(self, item_id):
        """
        get all users who rated item item_id and get the corresponding ratings
        """
        ids = np.where(self.Y_data_n[:,1] == item_id)[0] 
        user_ids = self.Y_data_n[ids, 0].astype(np.int32)
        ratings = self.Y_data_n[ids, 2]
        return (user_ids, ratings)
```

**Cập nhật \\(\mathbf{X}, \mathbf{W}\\):**

```python
    def updateX(self):
        for m in xrange(self.n_items):
            user_ids, ratings = self.get_users_who_rate_item(m)
            Wm = self.W[:, user_ids]
            # gradient
            grad_xm = -(ratings - self.X[m, :].dot(Wm)).dot(Wm.T)/self.n_ratings + \
                                               self.lam*self.X[m, :]
            self.X[m, :] -= self.learning_rate*grad_xm.reshape((self.K,))
    
    def updateW(self):
        for n in xrange(self.n_users):
            item_ids, ratings = self.get_items_rated_by_user(n)
            Xn = self.X[item_ids, :]
            # gradient
            grad_wn = -Xn.T.dot(ratings - Xn.dot(self.W[:, n]))/self.n_ratings + \
                        self.lam*self.W[:, n]
            self.W[:, n] -= self.learning_rate*grad_wn.reshape((self.K,))
```

**Phần thuật toán chính:**

```python
    def fit(self):
        self.normalize_Y()
        for it in xrange(self.max_iter):
            self.updateX()
            self.updateW()
            if (it + 1) % self.print_every == 0:
                rmse_train = self.evaluate_RMSE(self.Y_raw_data)
                print 'iter =', it + 1, ', loss =', self.loss(), ', RMSE train =', rmse_train
```

**Dự đoán:**

```python    
    def pred(self, u, i):
        """ 
        predict the rating of user u for item i 
        if you need the un
        """
        u = int(u)
        i = int(i)
        if self.user_based:
            bias = self.mu[u]
        else: 
            bias = self.mu[i]
        pred = self.X[i, :].dot(self.W[:, u]) + bias 
        # truncate if results are out of range [0, 5]
        if pred < 0:
            return 0 
        if pred > 5: 
            return 5 
        return pred 
        
    
    def pred_for_user(self, user_id):
        """
        predict ratings one user give all unrated items
        """
        ids = np.where(self.Y_data_n[:, 0] == user_id)[0]
        items_rated_by_u = self.Y_data_n[ids, 1].tolist()              
        
        y_pred = self.X.dot(self.W[:, user_id]) + self.mu[user_id]
        predicted_ratings= []
        for i in xrange(self.n_items):
            if i not in items_rated_by_u:
                predicted_ratings.append((i, y_pred[i]))
        
        return predicted_ratings
```

**Đánh giá kết quả bằng cách đo Root Mean Square Error:**

```python    
    def evaluate_RMSE(self, rate_test):
        n_tests = rate_test.shape[0]
        SE = 0 # squared error
        for n in xrange(n_tests):
            pred = self.pred(rate_test[n, 0], rate_test[n, 1])
            SE += (pred - rate_test[n, 2])**2 

        RMSE = np.sqrt(SE/n_tests)
        return RMSE
```


<a name="-ap-dung-len-movielens-k"></a>

### 3.2. Áp dụng lên MovieLens 100k

Chúng ta cùng quay lại với cơ sở dữ liệu [MovieLens 100k](/2017/05/17/contentbasedrecommendersys/#-co-so-du-lieu-movielens-k)
```python
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

ratings_base = pd.read_csv('ml-100k/ub.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('ml-100k/ub.test', sep='\t', names=r_cols, encoding='latin-1')

rate_train = ratings_base.as_matrix()
rate_test = ratings_test.as_matrix()

# indices start from 0
rate_train[:, :2] -= 1
rate_test[:, :2] -= 1
```

Kết quả nếu sư dụng cách **chuẩn hoá dựa trên _user_:**

```python
rs = MF(rate_train, K = 10, lam = .1, print_every = 10, 
    learning_rate = 0.75, max_iter = 100, user_based = 1)
rs.fit()
# evaluate on test data
RMSE = rs.evaluate_RMSE(rate_test)
print '\nUser-based MF, RMSE =', RMSE
```

    iter = 10 , loss = 5.67288309116 , RMSE train = 1.20479476967
    iter = 20 , loss = 2.64823713338 , RMSE train = 1.03727078113
    iter = 30 , loss = 1.34749564429 , RMSE train = 1.02937828335
    iter = 40 , loss = 0.754769340402 , RMSE train = 1.0291792473
    iter = 50 , loss = 0.48310745143 , RMSE train = 1.0292035212
    iter = 60 , loss = 0.358530096403 , RMSE train = 1.02921183102
    iter = 70 , loss = 0.30139979707 , RMSE train = 1.02921377947
    iter = 80 , loss = 0.27520033847 , RMSE train = 1.02921421055
    iter = 90 , loss = 0.263185542009 , RMSE train = 1.02921430477
    iter = 100 , loss = 0.257675693217 , RMSE train = 1.02921432529
    
    User-based MF, RMSE = 1.06037991127

Ta nhận thấy rằng giá trị `loss` giảm dần và `RMSE train` cũng giảm dần khi số vòng lặp tăng lên. RMSE có cao hơn so với Neighborhood-based Collaborative Filtering (~0.99) một chút nhưng vẫn tốt hơn kết quả của Content-based Recommendation Systems rất nhiều (~1.2).

Nếu **chuẩn hoá dựa trên _item_:**

```python
rs = MF(rate_train, K = 10, lam = .1, print_every = 10, learning_rate = 0.75, max_iter = 100, user_based = 0)
rs.fit()
# evaluate on test data
RMSE = rs.evaluate_RMSE(rate_test)
print '\nItem-based MF, RMSE =', RMSE
```

    iter = 10 , loss = 5.62978100103 , RMSE train = 1.18231933756
    iter = 20 , loss = 2.61820113008 , RMSE train = 1.00601013825
    iter = 30 , loss = 1.32429630221 , RMSE train = 0.996672160644
    iter = 40 , loss = 0.734890958031 , RMSE train = 0.99621264651
    iter = 50 , loss = 0.464793412146 , RMSE train = 0.996184081495
    iter = 60 , loss = 0.340943058213 , RMSE train = 0.996181347407
    iter = 70 , loss = 0.284148579208 , RMSE train = 0.996180972472
    iter = 80 , loss = 0.258103818785 , RMSE train = 0.996180914097
    iter = 90 , loss = 0.246160195903 , RMSE train = 0.996180905172
    iter = 100 , loss = 0.240683073898 , RMSE train = 0.996180903957
    
    Item-based MF, RMSE = 1.04980475198

Kết quả có tốt hơn một chút. 

Chúng ta cùng làm thêm một thí nghiệm nữa khi không sử dụng regularization, tức `lam = 0`:


```python
rs = MF(rate_train, K = 2, lam = 0, print_every = 10, learning_rate = 1, max_iter = 100, user_based = 0)
rs.fit()
# evaluate on test data
RMSE = rs.evaluate_RMSE(rate_test)
print '\nItem-based MF, RMSE =', RMSE
```

Nếu các bạn chạy đoạn code trên, các bạn sẽ thấy chất lượng của mô hình giảm đi rõ rệt (RMSE cao).

<a name="-ap-dung-len-movielens-m"></a>

### 3.3. Áp dụng lên MovieLens 1M

Tiếp theo, chúng ta cùng đến với một bộ cơ sở dữ liệu lớn hơn là [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) bao gồm xấp xỉ 1 triệu _ratings_ của 6000 người dùng lên 4000 bộ phim. Đây là một bộ cơ sở dữ liệu lớn, thời gian _training_ cũng sẽ tăng theo. Bạn đọc cũng có thể thử áp dụng mô hình Neighborhood-based Collaborative Filtering lên cơ sở dữ liệu này để so sánh kết quả. Tôi dự đoán là thời gian _training_ sẽ nhanh nhưng thời gian _inference_ sẽ rất lâu. 

**Load dữ liệu:**

```python
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

ratings_base = pd.read_csv('ml-1m/ratings.dat', sep='::', names=r_cols, encoding='latin-1')
ratings = ratings_base.as_matrix()

# indices in Python start from 0
ratings[:, :2] -= 1
```

**Tách tập training và test, sử dụng 1/3 dữ liệu cho test**

```python
from sklearn.model_selection import train_test_split

rate_train, rate_test = train_test_split(ratings, test_size=0.33, random_state=42)
print X_train.shape, X_test.shape
```

    (670140, 4) (330069, 4)

**Áp dụng Matrix Factorization:**

```python
rs = MF(rate_train, K = 2, lam = 0.1, print_every = 2, learning_rate = 2, max_iter = 10, user_based = 0)
rs.fit()
# evaluate on test data
RMSE = rs.evaluate_RMSE(rate_test)
print '\nItem-based MF, RMSE =', RMSE
```

    iter = 2 , loss = 6.80832412832 , RMSE train = 1.12359545594
    iter = 4 , loss = 4.35238943299 , RMSE train = 1.00312745587
    iter = 6 , loss = 2.85065420416 , RMSE train = 0.978490200028
    iter = 8 , loss = 1.90134941041 , RMSE train = 0.974189487594
    iter = 10 , loss = 1.29580344305 , RMSE train = 0.973438724579
    
    Item-based MF, RMSE = 0.981631017423

Kết quả khá ấn tượng sau 10 vòng lặp. Kết quả khi áp dụng Neighborhood-based Collaborative Filtering là khoảng 0.92 nhưng thời gian _inference_ khá lớn. 

<a name="-thao-luan"></a>

## 4. Thảo luận

<a name="-khi-co-bias"></a>

### 4.1. Khi có bias
Một lợi thế của hướng tiếp cận Matrix Factorization cho Collaborative Filtering là khả năng linh hoạt của nó khi có thêm các điều kiện ràng buộc khác, các điều kiện này có thể liên quan đến quá trình xử lý dữ liệu hoặc đến từng ứng dụng cụ thể. 

Giả sử ta chưa chuẩn hoá các giá trị _ratings_ mà sử dụng trực tiếp giá trị ban đầu của chúng trong đẳng thức \\((1)\\). Việc chuẩn hoá cũng có thể được tích hợp trực tiếp vào trong hàm mất mát. Như tôi đã đề cập, các _ratings_ thực tế đều có những thiên lệch về _users_ hoặc/và _items_. Có _user_ dễ và khó tính, cũng có những _item_ được _rated_ cao hơn những _items_ khác chỉ vì _user_ thấy các _users_ khác đã đánh giá _item_ đó cao rồi. Vấn đề thiên lệch có thể được giải quyết bằng các biến gọi là _biases_, phụ thuộc vào mỗi _user_ và _item_ và có thể được tối ưu cùng với \\(\mathbf{X}\\) và \\(\mathbf{W}\\). Khi đó, _ratings_ của _user_ \\(n\\) lên _item_ \\(m\\) không chỉ được xấp xỉ bằng \\(\mathbf{x}\_m\mathbf{w}\_n\\) mà còn phụ thuộc vào các _biases_ của _item_ \\(m\\) và _user_ \\(n\\) nữa. Ngoài ra, giá trị này cũng có thể phụ thuộc vào giá trị trung bình của toàn bộ _ratings_ nữa: 

\\[
y\_{mn} \approx \mathbf{x}\_m \mathbf{w}\_n + b_m + d_n + \mu
\\]

với \\(b_m, d_n, \mu\\) lần lượt là bias của _item_ \\(m\\), _user_ \\(n\\), và giá trị trung bình của toàn bộ các _ratings_ (là hằng số). 

Lúc này, hàm mất mát có thể được thay đổi thành: 
\\[
\begin{eqnarray}
\mathcal{L}(\mathbf{X}, \mathbf{W}, \mathbf{b}, \mathbf{d}) &=& \frac{1}{2s} \sum_{n=1}^N \sum_{m:r_{mn} = 1} (\mathbf{x}\_m\mathbf{w}\_n + b_m + d_n +\mu - y\_{mn})^2 + \\\ 
&& + \frac{\lambda}{2} (\|\|\mathbf{X}\|\|_F^2 + \|\|\mathbf{W}\|\|_F^2 + \|\|\mathbf{b}\|\|_2^2  + \|\|\mathbf{d}\|\|_2^2)
\end{eqnarray}
\\]

Việc tính toán đạo hàm cho từng biến không quá phức tạp, tôi sẽ không bàn tiếp ở đây. Tuy nhiên, nếu bạn quan tâm, bạn có thể tham khảo [source code mà tôi viết tại đây](https://github.com/tiepvupsu/tiepvupsu.github.io/tree/master/assets/25_mf/python). Link này cũng kèm theo các ví dụ nêu trong Mục 3 và dữ liệu liên quan. 


<a name="-nonnegative-matrix-factorization"></a>

### 4.2. Nonnegative Matrix Factorization
Khi dữ liệu chưa được chuẩn hoá, chúng đều mang các giá trị không âm. Nếu dải giá trị của _ratings_ có chứa giá trị âm, ta chỉ cần cộng thêm vào Utility Matrix một giá trị hợp lý để có được các _ratings_ là các số không âm. Khi đó, một phương pháp Matrix Factorization khác cũng được sử dụng rất nhiều và mang lại hiệu quả cao trong Recommendation Systems là Nonnegative Matrix Factorization, tức phân tích ma trận thành tích các ma trận có các phần tử không âm. 

Bằng Matrix Factorization, các _users_ và _items_ được liên kết với nhau bởi các _latent features_ (tính chất ẩn). Độ liên kết của mỗi _user_ và _item_ tới mỗi latent feature được đo bằng thành phần tương ứng trong feature vector hệ số của chúng, giá trị càng lớn thể hiện việc _user_ hoặc _item_ có liên quan đến latent feature đó càng lớn. Bằng trực giác, sự liên quan của một _user_ hoặc _item_ đến một latent feature nên là một số không âm với giá trị 0 thể hiện việc _không liên quan_. Hơn nữa, mỗi _user_ và _item_ chỉ _liên quan_ đến một vài _latent feature_ nhất định. Vậy nên feature vectors cho _users_ và _items_ nên là các vectors không âm và có rất nhiều giá trị bằng 0. Những nghiệm này có thể đạt được bằng cách cho thêm ràng buộc không âm vào các thành phần của \\(\mathbf{X}\\) và \\(\mathbf{W}\\).

Bạn đọc muốn tìm hiểu thêm về Nonnegative Matrix Factorization có thể tham khảo các tài liệu ở Mục 6. 

<a name="-incremental-matrix-factorization"></a>

### 4.3. Incremental Matrix Factorization 
Như đã đề cập, thời gian _inference_ của Recommendation Systems sử dụng Matrix Factorization là rất nhanh nhưng thời gian _training_ là khá lâu với các tập dữ liệu lớn. Thực tế cho thấy, Utility Matrix thay đổi liên tục vì có thêm _users_, _items_ cũng như các _ratings_ mới hoặc _user_ muốn thay đổi _ratings_ của họ, vì vậy hai ma trận \\(\mathbf{X}\\) và \\(\mathbf{W}\\) phải thường xuyên được cập nhật. Điều này đồng nghĩa với việc ta phải tiếp tục thực hiện quá trình _training_ vốn tốn khá nhiều thời gian. 

Việc này được giải quyết phần nào bằng Incremental Matrix Factorization. Bạn đọc quan tâm có thể đọc [Fast incremental matrix factorization for recommendation with positive-only feedback](https://ai2-s2-pdfs.s3.amazonaws.com/c827/d2267640a7a913250fa5046a16ff078a5ce4.pdf).

<a name="-others"></a>

### 4.4. Others

* Bài toán Matrix Factorization có nhiều hướng giải quyết khác ngoài Gradient Descent. Bạn đọc có thể xem thêm [Alternating Least Square (ALS)](https://stanford.edu/~rezab/classes/cme323/S15/notes/lec14.pdf), [Generalized Low Rank Models](https://stanford.edu/~rezab/classes/cme323/S15/notes/lec14.pdf). Trong bài tiếp theo, tôi sẽ viết về [Singular Value Decomposition (SVD)](https://en.wikipedia.org/wiki/Singular_value_decomposition), một phương pháp phổ biến trong Matrix Factorization,  được sử dụng không những trong (Recommendation) Systems mà còn trong nhiều hệ thống khác. Mời các bạn đón đọc. 

* [Source code](https://github.com/tiepvupsu/tiepvupsu.github.io/tree/master/assets/25_mf/python)
<a name="-tai-lieu-tham-khao"></a>

## 6. Tài liệu tham khảo

[1] [Recommendation Systems - Stanford InfoLab](http://infolab.stanford.edu/~ullman/mmds/ch9.pdf)

[2] [Collaborative Filtering - Stanford University](https://www.youtube.com/watch?v=h9gpufJFF-0&t=436s)

[3] [Recommendation systems - Machine Learning - Andrew Ng](https://www.youtube.com/watch?v=saXRzxgFN0o&list=PL_npY1DYXHPT-3dorG7Em6d18P4JRFDvH)

[4] Ekstrand, Michael D., John T. Riedl, and Joseph A. Konstan. "[Collaborative filtering recommender systems.](http://herbrete.vvv.enseirb-matmeca.fr/IR/CF_Recsys_Survey.pdf)" Foundations and Trends® in Human–Computer Interaction 4.2 (2011): 81-173.

[5] [Matrix factorization techniques for recommender systems](https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf)

[6] [Matrix Factorization For Recommender Systems](http://joyceho.github.io/cs584_s16/slides/mf-16.pdf)

[7] [Learning from Incomplete Ratings Using Non-negative Matrix Factorization](http://www.siam.org/meetings/sdm06/proceedings/059zhangs2.pdf)

[8] [Fast Incremental Matrix Factorization for Recommendation with Positive-Only Feedback](https://ai2-s2-pdfs.s3.amazonaws.com/c827/d2267640a7a913250fa5046a16ff078a5ce4.pdf)
