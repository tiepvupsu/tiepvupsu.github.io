---
layout: post
comments: true
title:  "Bài 16: Convex sets và convex functions"
title2:  "16. Convex sets và convex functions"
date:   2017-03-12 15:22:00
permalink: 2017/03/12/convexity/
mathjax: true
tags: Convex Optimization
category: Optimization
sc_project: 11281831
sc_security: f2dfc7eb
img: \assets\16_convexity\norm2_surf.png
summary: Giới thiệu về tập hợp lồi và hàm số lồi trong Toán Tối Ưu.
---

Bài này có khá nhiều khái niệm mới, mong bạn đọc thông cảm khi tôi sử dụng các khái niệm này ở cả tiếng Anh và tiếng Việt.

_Bài chủ yếu nói về toán, nếu bạn đọc không hiểu ngay cũng không sao, ngày đầu tôi làm quen với những khái niệm này cũng không thể hấp thụ được ngay. Làm nhiều, đọc nhiều rồi sẽ ngấm dần._

Bạn đọc có thể xem bản pdf [tại đây](https://github.com/tiepvupsu/tiepvupsu.github.io/blob/master/assets/latex/book_CVX.pdf).

<!-- MarkdownTOC -->

- [1. Giới thiệu](#-gioi-thieu)
- [2. Convex sets](#-convex-sets)
    - [2.1. Định nghĩa](#-dinh-nghia)
    - [2.2. Ví dụ](#-vi-du)
        - [2.2.1. Hyperplanes và halfspaces](#-hyperplanes-va-halfspaces)
        - [2.2.2. Norm balls](#-norm-balls)
    - [2.3. Giao của các tập lồi là một tập lồi.](#-giao-cua-cac-tap-loi-la-mot-tap-loi)
    - [2.4. Convex combination và Convex hulls](#-convex-combination-va-convex-hulls)
- [3. Convex functions](#-convex-functions)
    - [3.1. Định nghĩa](#-dinh-nghia-1)
    - [3.2. Các tính chất cơ bản](#-cac-tinh-chat-co-ban)
    - [3.3. Ví dụ](#-vi-du-1)
        - [3.3.1. Các hàm một biến](#-cac-ham-mot-bien)
        - [3.3.3. Affine functions](#-affine-functions)
        - [3.3.3. Quadratic forms](#-quadratic-forms)
        - [3.3.4. Norms](#-norms)
    - [3.4. \\\(\alpha-\\\) sublevel sets](#-\\\alpha-\\-sublevel-sets)
    - [3.5. Kiểm tra tính chất lồi dựa vào đạo hàm.](#-kiem-tra-tinh-chat-loi-dua-vao-dao-ham)
        - [3.5.1. First-order condition](#-first-order-condition)
        - [3.5.2. Second-order condition](#-second-order-condition)
- [4. Tóm tắt](#-tom-tat)
- [5. Tài liệu tham khảo](#-tai-lieu-tham-khao)

<!-- /MarkdownTOC -->


<a name="-gioi-thieu"></a>

## 1. Giới thiệu
Từ đầu đến giờ, chúng ta đã làm quen với rất nhiều bài toán tối ưu. Học Machine Learning là phải học Toán Tối Ưu, và để hiểu hơn về Toán Tối Ưu, với tôi cách tốt nhất là tìm hiểu các thuật toán Machine Learning. Cho tới lúc này, những bài toán tối ưu các bạn đã nhìn thấy trong blog đều là các bài toán tối ưu không ràng buộc (unconstrained optimization problems), tức tối ưu hàm mất mát mà không có điều kiện ràng buộc (constraints) nào về nghiệm cả.

Không chỉ trong Machine Learning, trên thực tế các bài toán tối ưu thường có rất nhiều ràng buộc khác nhau. Ví dụ:

* Tôi muốn thuê một ngôi nhà cách trung tâm Hà Nội không quá 5km với giá càng thấp càng tốt. Trong bài toán này, giá thuê nhà chính là hàm mất mát (_loss function_, đôi khi người ta cũng dùng _cost function_ để chỉ hàm số cần tối ưu), điều kiện khoảng cách không quá 5km chính là ràng buộc (constraint).

* Quay lại [bài toán dự đoán giá nhà theo Linear Regression](/2016/12/28/linearregression/#-gioi-thieu), giá nhà là một hàm tuyến tính của diện tích, số phòng ngủ và khoảng cách tới trung tâm. Rõ ràng, khi làm bài toán này, ta dự đoán rằng giá nhà tăng theo diện tích và số phòng ngủ, giảm theo khoảng cách. Vậy nên một nghiệm được gọi là _có lý một chút_ nếu hệ số tương ứng với diện tích và số phòng ngủ là dương, hệ số tương ứng với khoảng cách là âm. Để tránh các nghiệm ngoại lai không mong muốn, khi giải bài toán tối ưu, ta nên cho thêm các điều kiện ràng buộc này.

Trong Tối Ưu, một bài toán có ràng buộc thường được viết dưới dạng:

\\[
\begin{eqnarray}
\mathbf{x}^* &=& \arg\min_{\mathbf{x}} f_0(\mathbf{x})\\\
\text{subject to:}~ && f_i(\mathbf{x}) \leq 0, ~~ i = 1, 2, \dots, m \\\
&& h_j(\mathbf{x}) = 0, ~~ j = 1, 2, \dots, p
\end{eqnarray}
\\]

Trong đó, vector \\(\mathbf{x} = [x_1, x_2, \dots, x_n]^T\\) được gọi là _biến tối ưu_ (_optimization variable_). Hàm số \\(f_0: \mathbb{R}^n \rightarrow \mathbb{R}\\) được gọi là _hàm mục tiêu_ (_objective function_, các hàm mục tiêu trong Machine Learning thường được gọi là _hàm mất mát_). Các hàm số \\(f_i, h_j: \mathbb{R}^n \rightarrow \mathbb{R}, i = 1, 2, \dots, m; j = 1, 2, \dots, p\\) được gọi là các _hàm ràng buộc_ (hoặc đơn giản là _ràng buộc_ - constraints). Tập hợp các điểm \\(\mathbf{x}\\) thỏa mãn các _ràng buộc_ được gọi là _feasible set_. Mỗi điểm trong _feasible set_ được gọi là _feasible point_, các điểm không trong _feasible set_ được gọi là _infeasible points_.

**Chú ý:**

* Nếu bài toán là tìm giá trị lớn nhất thay vì nhỏ nhất, ta chỉ cần đổi dấu của \\(f_0(\mathbf{x})\\).

* Nếu ràng buộc là _lớn hơn hoặc bằng_, tức \\(f_i(\mathbf{x}) \geq b_i\\), ta chỉ cần đổi dấu của ràng buộc là sẽ có điều kiện _nhỏ hơn hoặc bằng_ \\(-f_i(\mathbf{x}) \leq -b_i\\).

* Các ràng buộc cũng có thể là _lớn hơn_ hoặc _nhỏ hơn_.

* Nếu ràng buộc là _bằng nhau_, tức \\(h_j(\mathbf{x}) = 0\\), ta có thể viết nó dưới dạng hai bất đẳng thức \\(h_j(\mathbf{x}) \leq 0\\) và \\(-h_j(\mathbf{x}) \leq 0\\). Trong một vài tài liệu, người ta bỏ các phương trình ràng buộc \\(h_j(\mathbf{x})= 0\\) đi.

* Trong bài viết này, \\(\mathbf{x}, \mathbf{y}\\) được dùng chủ yếu để ký hiệu các biến số, không phải là dữ liệu như trong các bài trước. Biến tối ưu chính là biến được ghi dưới dấu \\(\arg \min\\). Khi viết một bài toán Tối Ưu, ta cần chỉ rõ biến nào cần được tối ưu, biến nào là cố định.

Các bài toán tối ưu, nhìn chung không có cách giải tổng quát, thậm chí có những bài chưa có lời giải. Hầu hết các phương pháp tìm nghiệm không chứng minh được nghiệm tìm được có phải là _global optimal_ hay không, tức đúng là điểm làm cho hàm số đạt giá trị nhỏ nhất hay lớn nhất hay không. Thay vào đó, nghiệm thường là các _local optimal_, tức các _điểm cực trị_.

Để bắt đầu học Tối Ưu, chúng ta cần học một mảng rất quan trọng trong đó, có tên là _Tối Ưu Lồi_ (convex optimization), trong đó _hàm mục tiêu_ là một _hàm lồi_ (convex function), _feasible set_ là một _tập lồi_ (convex set). Những tính chất đặc biệt về _local optimal_ và _global optimal_ của một _hàm lồi_ khiến Tối Ưu Lồi trở nên cực kỳ quan trọng. Trong bài viết này, tôi sẽ giới thiệu tới các bạn các định nghĩa và tính chất cơ bản của _tập lồi_ và _hàm lồi_. _Bài toán tối ưu lồi_ (convex optimization problems) sẽ được đề cập trong bài tiếp theo.

<a name="-convex-sets"></a>

## 2. Convex sets

<a name="-dinh-nghia"></a>

### 2.1. Định nghĩa
Khái niệm về _convex sets_ có lẽ không xa lạ với các bạn học sinh Việt Nam khi chúng ta đã nghe về _đa giác lồi_. _Lồi_, hiểu đơn giản là _phình ra ngoài_, hoặc _nhô ra ngoài_. Trong toán học, _bằng phẳng_ cũng được coi là _lồi_.

**Định nghĩa 1:** Một tập hợp được gọi là _tập lồi_ (convex set) nếu đoạn thẳng nối hai điểm _bất kỳ_ trong tập hợp hợp đó nằm trọn vẹn trong tập hợp đó.

Một vài ví dụ về convex sets:
<hr>
<div class="imgcap">
 <img src ="/assets/16_convexity/convexsets.png" align = "center" width = "800">
 <div class = "thecap">Hình 1: Các ví dụ về convex sets.</div>
</div>
<hr>

Các hình với đường biên màu đen thể hiện việc bao gồm cả biên, biên màu trắng thể hiện việc biên đó không nằm trong tập hợp đang xét. Đường hoặc đoạn thằng cũng là một tập lồi theo định nghĩa phía trên.

Một vài ví dụ thực tế:

* Giả sử có một căn phòng có dạng hình _lồi_, nếu ta đặt một bóng đèn đủ sáng ở bất kỳ vị trí nào trong phòng, mọi điểm trong căn phòng đều được chiếu sáng.

* Nếu một đất nước có bản đồ dạng một hình _lồi_ thì đường bay nối giữa hai thành phố bất kỳ trong đất nước đó đều nằm trọn vẹn trong không phận của nước đó. (Không như Việt Nam, muốn bay thẳng Hà Nội - Hồ Chí Minh phải bay qua không phận Campuchia).

Dưới đây là một vài ví dụ về _nonconvex sets_, tức tập hợp mà không phải là lồi:

<hr>
<div class="imgcap">
 <img src ="/assets/16_convexity/nonconvexsets.png" align = "center" width = "800">
 <div class = "thecap">Hình 2: Các ví dụ về nonconvex sets.</div>
</div>
<hr>

Ba hình đầu tiên không phải là lồi vì các đường nét đứt chứa nhiều điểm không nằm trong các tập đó. Hình thứ tư, hình vuông không có biên ở đáy, không phải là _tập lồi_ vì đoạn thẳng nối hai điểm ở đáy có thể chứa phần ở giữa không thuộc tập đang xét (Nếu không có biên thì thình vuông vẫn là một _tập lồi_, nhưng biên nửa vời như ví dụ này thì hãy chú ý). Một đường cong bất kỳ cũng không phải là _tập lồi_ vì dễ thấy đường thẳng nối hai điểm bất kỳ không thuộc đường cong đó.

Để mô tả một _tập lồi_ dưới dạng toán học, ta sử dụng:

**Định nghĩa 2:** Một tập hợp \\(\mathcal{C}\\) được gọi là _convex_ nếu với hai điểm bất kỳ \\(\mathbf{x}\_1, \mathbf{x}\_2 \in \mathcal{C}\\), điểm \\( \mathbf{x}\_{\theta} = \theta \mathbf{x}_1 + (1 - \theta) \mathbf{x}_2\\) cũng nằm trong \\(\mathcal{C}\\) với bất kỳ \\(0 \leq \theta \leq 1\\).

Có thể thấy rằng, tập hợp các điểm có dạng \\(\left(\theta \mathbf{x}\_1 + (1 - \theta) \mathbf{x}\_2\right)\\) chính là _đoạn thẳng_ nối hai điểm \\(\mathbf{x}_1\\) và \\(\mathbf{x}_2\\).

Với các định nghĩa này thì _toàn bộ không gian_ là một _tập lồi_ vì đoạn thằng nào cũng nằm trong không gian đó. Tập rỗng cũng có thể coi là một trường hợp đặc biệt của _tập lồi_.

Dưới đây là một vài ví dụ hay gặp về _tập lồi_.
<a name="-vi-du"></a>

### 2.2. Ví dụ
<a name="-hyperplanes-va-halfspaces"></a>

#### 2.2.1. Hyperplanes và halfspaces
Một **hyperplane** (siêu mặt phẳng) trong không gian \\(n\\) chiều là tập hợp các điểm thỏa mãn phương trình:
\\[
a_1 x_1 + a_2 x_2 + \dots + a_n x_n = \mathbf{a}^T\mathbf{x} = b
\\]
với \\(b, a_i, i = 1, 2, \dots, n\\) là các số thực.

Hyperplanes là các _tập lồi_. Điều này có thể dễ dàng suy ra từ Định nghĩa 1. Với Định nghĩa 2, chúng ta cũng dễ dàng nhận thấy. Nếu:

\\[
\mathbf{a}^T\mathbf{x}\_1 = \mathbf{a}^T\mathbf{x}\_2 = b
\\]
thì với \\(0 \leq \theta \leq 1\\) bất kỳ:
\\[
\mathbf{a}^T\mathbf{x}_{\theta} = \mathbf{a}^T(\theta \mathbf{x}_1 + (1 - \theta)\mathbf{x}_2)) = \theta b + (1 - \theta) b  = b
\\]

Một **halfspace** (nửa không gian) trong không gian \\(n\\) chiều là tập hợp các điểm thỏa mãn bất phương trình:
\\[
a_1 x_1 + a_2 x_2 + \dots + a_n x_n = \mathbf{a}^T\mathbf{x} \leq b
\\]
với \\(b, a_i, i = 1, 2, \dots, n\\) là các số thực.

Các halfspace cũng là các tập lồi, bạn đọc có thể dễ dàng nhận thấy theo Định nghĩa 1 hoặc chứng minh theo Định nghĩa 2.

<a name="-norm-balls"></a>

#### 2.2.2. Norm balls
**Euclidean balls** (hình tròn trong mặt phẳng, hình cầu trong không gian ba chiều) là tập hợp các điểm có dạng:
\\[
B(\mathbf{x}_c, r) = \\{\mathbf{x} ~\big|~ \|\|\mathbf{x} - \mathbf{x}_c\|\|_2 \leq r \\} = \\{\mathbf{x}_c + r\mathbf{u} ~\big|~ \|\|\mathbf{u}\|\|_2 \leq 1\\}
\\]

Theo Định nghĩa 1, chúng ta có thể _thấy_ Euclidean balls là các tập lồi, nếu phải chứng minh, ta dùng Định nghĩa 2 và [các tính chất của norms](/math/#-norms-chuan). Với \\(\mathbf{x}_1, \mathbf{x}_2\\) bất kỳ thuộc \\(B(\mathbf{x}_c, r)\\) và \\(0 \leq \theta \leq 1\\) bất kỳ:
\\[
\begin{eqnarray}
\|\|\mathbf{x}\_{\theta} - \mathbf{x}_c\|\|_2 &=& \|\|\theta(\mathbf{x}_1 - \mathbf{x}_c)  + (1 - \theta) (\mathbf{x}_2 - \mathbf{x}_c)\|\|_2 \\\
&\leq& \theta \|\|\mathbf{x}_1 - \mathbf{x}_c\|\|_2 + (1 - \theta)\|\|\mathbf{x}_2 - \mathbf{x}_c\|\|_2 \\\
&\leq& \theta r + ( 1 - \theta) r = r
\end{eqnarray}
\\]

Vậy nên \\(\mathbf{x}\_{\theta} \in B(\mathbf{x}_c, r)\\).

**Euclidean ball** sử dụng norm 2 làm khoảng cách. Nếu sử dụng norm bất kỳ là khoảng cách, ta vẫn được một _tập lồi_.

**Khi sử dụng norm p:**
\\[
\|\|\mathbf{x}\|\|\_p = (\|x_1\|^p + \|x_2\|^p + \dots \|x_n\|^p)^{\frac{1}{p}} ~~(1)
\\]
với **p là một số thực bất kỳ không nhỏ hơn 1** ta cũng thu được các _tập lồi_.

Hình dưới đây minh họa tập hợp các điểm có tọa độ \\((x, y)\\) trong không gian hai chiều thỏa mãn:
\\[
(|x|^p + |y|^p)^{1/p} \leq 1 ~~~(1)
\\]
với hàng trên là các tập với \\(0 < p < 1\\) (không phải norm) và hàng dưới tương ứng với \\(p \geq 1\\):
<hr>
<div class="imgcap">
 <img src ="/assets/16_convexity/normballs.png" align = "center" width = "800">
 <div class = "thecap">Hình 3. Hình dạng của các tập hợp bị chặn bởi pseudo-norms (hàng trên) và norm (hàng dưới).</div>
</div>
<hr>
Chúng ta có thể thấy rằng khi \\(p\\) nhỏ gần bằng 0, tập hợp các điểm thỏa mãn bất đẳng thức (1) gần như nằm trên các trục tọa độ và bị chặn trong đoạn \\([0, 1]\\). Quan sát này sẽ giúp ích cho các bạn khi làm việc với (giả) norm 0 sau này. Khi \\(p \rightarrow \infty\\), các tập hợp hội tụ về hình vuông.

Đây cũng là một trong các lý do vì sao cần có điều kiện \\(p \geq 1\\) khi định nghĩa norm.

**Ellipsoids**

Các ellipsoids (ellipse trong không gian nhiều chiều) cũng là các _tập lồi_. Thực chất, ellipsoides có mối quan hệ mật thiết tới [Khoảng cách Mahalanobis](https://en.wikipedia.org/wiki/Mahalanobis_distance). Khoảng cách này vốn dĩ là một norm nên ta có thể chứng minh theo Định nghĩa 2 được tính chất lồi của các ellipsoids.

**Mahalanobis norm** của một vector \\(\mathbf{x} \in \mathbb{R}^n\\) được định nghĩa là:
\\[
\|\|\mathbf{x}\|\|_{\mathbf{A}} = \sqrt{\mathbf{x}^T\mathbf{A}^{-1}\mathbf{x}}
\\]

Với \\(\mathbf{A}\\) là một ma trận thỏa mãn:
\\[
\mathbf{x}^T\mathbf{A}^{-1}\mathbf{x} \geq 0, ~~\forall \mathbf{x} \in \mathbb{R}^n ~~ (2)
\\]
Khi một ma trận \\(\mathbf{A}\\) thỏa mãn điều kiện \\((2)\\), ta nói ma trận đó *xác định dương* (*positive definite*). 
<a name="positive-semidefinite"></a>
Nhân tiện, một ma trận \\(\mathbf{B}\\) được gọi là **nửa** *xác định dương* (*positive semidefinite*) nếu các *trị riêng* của nó là không âm. Khi đó \\(\mathbf{x}^T \mathbf{Bx} \geq 0, \forall \mathbf{x}\\). Nếu dấu bằng xảy ra khi và chỉ khi \\(\mathbf{x} = 0\\) thì ta nói ma trận đó *xác định dương*. Trong biểu thức \\((2)\\), vì ma trận \\(\mathbf{A}\\) có nghịch đảo nên mọi *trị riêng* của nó phải khác không. Vì vậy, \\(\mathbf{A}\\) là một ma trận *xác định dương*.

Một ma trận \\(\mathbf{A}\\) là _xác định dương_ hoặc _nửa xác định dương_ sẽ được ký hiệu lần lượt như sau:
\\[
\mathbf{A} \succ 0, ~~~~~ \mathbf{A} \succeq 0.
\\]

Cũng lại nhân tiện, khoảng cách Mahalanobis có liên quan đến *khoảng cách từ một điểm tới một phân phối xác suất* (from a point to a distribution). 
<a name="-giao-cua-cac-tap-loi-la-mot-tap-loi"></a>

### 2.3. Giao của các tập lồi là một tập lồi.
Việc này có thể nhận dễ nhận thấy với Hình 4 (trái) dưới đây. Giao của hai trong ba hoặc cả ba tập lồi đều là các tập lồi.

Việc chứng minh việc này theo Định nghĩa 2 cũng không khó. Nếu \\(\mathbf{x}_1, \mathbf{x}_2\\) thuộc vào giao của các tập lồi, tức thuộc tất cả các tập lồi đã cho, thì \\(\theta\mathbf{x}_1 + (1 - \theta) \mathbf{x}_2)\\) cũng thuộc vào tất cả các tập lồi, tức thuộc vào giao của chúng!

<hr>
<div class="imgcap">
 <img src ="/assets/16_convexity/intersection.png" align = "center" width = "800">
 <div class = "thecap">Hình 4. Trái: Giao của các tập lồi là một tập lồi. Phải: giao của các hyperplanes và halfspaces là một tập lồi và được gọi là polyhedron (số nhiều là polyhedra).</div>
</div>
<hr>

Từ đó suy ra giao của các _halfspaces_ và các _hyperplanes_ cũng là một tập lồi. Trong không gian hai chiều, tập lồi này chính là _đa giác lồi_, trong không gian ba chiều, nó có tên là _đa diện lồi_.

Trong không gian nhiều chiều, giao của các *halfspaces* và *hyperplanes* được gọi là **polyhedra**.

Giả sử có \\(m\\) *halfspaces* và \\(p\\) *hyperplanes*. Mỗi một *haflspace*, theo như đã trình bày phía trên, có thể viết dưới dạng \\(\mathbf{a}_i^T\mathbf{x} \leq b_i, ~\forall i = 1, 2, \dots, m\\). Mỗi một *hyperplane* có thể viết dưới dạng: \\(\mathbf{c}_i^T\mathbf{x} = d_i, ~\forall i = 1, 2, \dots, p\\).

 Vậy nếu đặt \\(\mathbf{A} = [\mathbf{a}_1, \mathbf{a}_2, \dots, \mathbf{a}_m]\\), \\(\mathbf{b} = [b_1, b_2, \dots, b_m]^T, \mathbf{C} = [\mathbf{c}_1, \mathbf{c}_2, \dots, \mathbf{c}_p]\\) và \\(\mathbf{d} = [d_1, d_2, \dots, d_p]^T\\), ta có thể viết polyhedra dưới dạng tập hợp các điểm \\(\mathbf{x}\\) thỏa mãn:
 \\[
 \mathbf{A}^T\mathbf{x} \preceq \mathbf{b}, ~~~~  \mathbf{C}^T\mathbf{x} = \mathbf{d}
 \\]
trong đó \\(\preceq\\) là *element-wise*, tức mỗi phần tử trong vế trái nhỏ hơn hoặc bằng phần tử tương ứng trong vế phải.

<a name="-convex-combination-va-convex-hulls"></a>

### 2.4. Convex combination và Convex hulls
Một điểm được gọi là **convex combination** (_tổ hợp lồi_) của các điểm \\(\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_k\\) nếu nó có thể viết dưới dạng:
\\[
\mathbf{x} = \theta_1 \mathbf{x}_1 + \theta_2 \mathbf{x}_2 + \dots  + \theta_k \mathbf{x}_k, ~~ \text{with} ~~ \theta_1 + \theta_2 + \dots + \theta_k = 1
\\]

**Convex hull** của một **tập hợp bất kỳ** là tập hợp tất cả các điểm là _convex combination_ của tập hợp đó. *Convex hull* là một _convex set_. *Convexhull* của một _convex set_ là chính nó. Một cách dễ nhớ, _convex hull_ của một tập hợp là một _convex set_ **nhỏ nhất** chứa tập hợp đó. Khái niệm **nhỏ nhất** rất khó định nghĩa, nhưng nó cũng là một cách nhớ trực quan.

Hai tập hợp được gọi là _linearly separable_ nếu các _convex hulls_ của chúng không có điểm chung.
<hr>
<div class="imgcap">
 <img src ="/assets/16_convexity/convex_hull.png" align = "center" width = "800">
 <div class = "thecap">Hình 5. Convex hull và Định lý separating hyperplane.</div>
</div>
<hr>
Trong hình trên, convex hull của các điểm màu xanh là vùng màu xám bao với các đa giác lồi. Ở hình bên phải thì vùng màu xám nằm dưới vùng màu xanh.

<hr>
**Separating hyperplane theorem:** Định lý này nói rằng nếu hai _tập lồi không rỗng_ \\(\mathcal{C}, \mathcal{D}\\) là _disjoint_ (không giao nhau), thì tồn tại vector \\(\mathbf{a}\\) và số \\(b\\) sao cho:
\\[
\mathbf{a}^T\mathbf{x} \leq b, \forall \mathbf{x} \in \mathcal{C}, ~~ \text{and}~~ \mathbf{a}^T\mathbf{x} \geq b, \forall \mathbf{x} \in \mathcal{D}
\\]
Tập hợp tất cả các điểm \\(\mathbf{x}\\) thỏa mãn \\(\mathbf{a}^T\mathbf{x} = b\\) chính là một hyperplane. Hyperplan này được gọi là _separating hyperplane_.
<hr>


Ngoài ra còn nhiều tính chất thú vị của các tập lồi và các phép toán bảo toàn chính chất _lồi_ của một tập hợp, các bạn được khuyến khích đọc thêm Chương 2 của cuốn Convex Optimization trong phần tài liệu tham khảo.


<a name="-convex-functions"></a>

## 3. Convex functions

Hẳn các bạn đã nghe tới khái niệm này khi ôn thi đại học môn toán. Khái niệm hàm lồi có quan hệ tới đạo hàm bậc hai và [Bất đẳng thức Jensen](https://vi.wikipedia.org/wiki/Bất_đẳng_thức_Jensen) (_nếu bạn chưa nghe tới phần này, không sao, bây giờ bạn sẽ biết_).

<a name="-dinh-nghia-1"></a>

### 3.1. Định nghĩa
Để trực quan, trước hết ta xem xét các hàm 1 biến, đồ thị của nó là một đường trong một mặt phẳng. Một hàm số được gọi là _lồi_ nếu **tập xác định của nó là một tập lồi** và nếu ta nối hai điểm bất kỳ trên đồ thị hàm số đó, ta được một đoạn thẳng nằm về phía trên hoặc nằm trên đồ thị (xem Hình 6).

Tập xác định (domain) của một hàm số \\(f(.)\\) thường được ký hiệu là \\(\text{dom} f\\).

Định nghĩa theo toán học:
<hr>
**Định nghĩa convex function:** Một hàm số \\(f: \mathbb{R}^n \rightarrow \mathbb{R} \\) được gọi là một _hàm lồi_ (convex function) nếu \\(\text{dom} f\\) là một _tập lồi_, và:
\\[
f(\theta\mathbf{x} + (1 - \theta) \mathbf{y}) \leq \theta f(\mathbf{x}) + (1 - \theta)f(\mathbf{y})
\\]
với mọi \\(\mathbf{x, y} \in \text{dom}f, 0 \leq \theta \leq 1\\).
<hr>
Điều kiện \\(\text{dom} f\\) là một _tập lồi_ là rất quan trọng, vì nếu không có nó, ta không định nghĩa được \\(f(\theta\mathbf{x} + (1 - \theta) \mathbf{y}) \\).

<hr>
<div class="imgcap">
 <img src ="/assets/16_convexity/convexf_def.png" align = "center" width = "500">
 <div class = "thecap">Hình 6. Convex function.</div>
</div>
<hr>

<a name="concave-function"></a>
Một hàm số \\(f\\) được gọi là **concave** (nếu bạn muốn dịch là _lõm_ cũng được, tôi không thích cách dịch này) nếu \\(-f\\) là **convex**. Một hàm số có thể không thuộc hai loại trên. Các hàm tuyến tính vừa *convex*, vừa *concave*.
<hr>
**Định nghĩa strictly convex function:** (tiếng Việt có một số tài liệu gọi là _hàm lồi mạnh_, _hàm lồi chặt_) Một hàm số \\(f: \mathbb{R}^n \rightarrow \mathbb{R} \\) được gọi là _strictly convex_  nếu \\(\text{dom} f\\) là một _tập lồi_, và:
\\[
f(\theta\mathbf{x} + (1 - \theta) \mathbf{y}) < \theta f(\mathbf{x}) + (1 - \theta)f(\mathbf{y})
\\]
với mọi \\(\mathbf{x, y} \in \text{dom}f, \mathbf{x} \neq \mathbf{y},  0 < \theta < 1\\).
<hr>
Tương tự với định nghĩa **strictly concave**.

Đây là một điểm quan trọng: **Nếu một hàm số là _strictly convex_ và có điểm cực trị, thì điểm cực trị đó là duy nhất và cũng là _global minimum_**.

<a name="-cac-tinh-chat-co-ban"></a>

### 3.2. Các tính chất cơ bản

* Nếu \\(f(\mathbf{x})\\) là _convex_ thì \\(af(\mathbf{x})\\) là _convex_ nếu \\(a > 0\\) và là _concave_ nếu \\(a < 0\\). Điều này có thể suy ra trực tiếp từ định nghĩa.

* Tổng của hai _hàm lồi_ là một _hàm lồi_, với tập xác định là giao của hai tập xác định kia (nhắc lại rằng giao của hai tập lồi là một tập lồi)

* **Pointwise maximum and supremum:** Nếu các hàm số \\(f_1, f_2, \dots, f_m\\) là _convex_ thì:
\\[
f(\mathbf{x}) = \max\\{f_1(\mathbf{x}), f_2(\mathbf{x}), \dots, f_m(\mathbf{x})\\}
\\]
cũng là _convex_ trên tập xác định là giao của tất cả các tập xác định của các hàm số trên. Hàm \\(\max\\) phía trên cũng có thể thay thế bằng [hàm \\(\text{sup}\\)](https://en.wikipedia.org/wiki/Infimum_and_supremum). Tính chất này có thể chứng minh được theo Định nghĩa. Bạn cũng có thể nhận ra dựa vào hình ví dụ dưới đây. Mọi đoạn thẳng nối hai điểm bất kì trên đường màu xanh đều _không nằm dưới_ đường màu xanh.

<hr>
<div class="imgcap">
 <img src ="/assets/16_convexity/max_point.png" align = "center" width = "400">
 <div class = "thecap">Hình 7. Ví dụ về Pointwise maximum.</div>
</div>
<hr>

<a name="-vi-du-1"></a>

### 3.3. Ví dụ
<a name="-cac-ham-mot-bien"></a>

#### 3.3.1. Các hàm một biến
**Các ví dụ về các _convex functions_ một biến:**

* Hàm \\( y = ax + b\\) là một _hàm lồi_ vì đường nối hai điểm bất kỳ nằm trên chính đồ thị đó.

* Hàm \\(y = e^{ax}\\) với \\(a \in \mathbb{R}\\) bất kỳ.

* Hàm \\(y = x^a\\) trên tập các số thực dương và \\(a \geq 1\\) hoặc \\(a \leq 0\\).

* Hàm _negative entropy_ \\(y = x \log x\\) trên tập các số thực dương.

Dưới đây là đồ thị của một vài _convex functions_:
<hr>
<div class="imgcap">
 <img src ="/assets/16_convexity/convexfunctions.png" align = "center" width = "800">
 <div class = "thecap">Hình 8. Ví dụ về các convex functions một biến.</div>
</div>
<hr>

**Các ví dụ về các _concave functions_ một biến:**

* Hàm \\(y = ax + b\\) là một _concave function_ vì \\(-y\\) là một _convex function_.

* Hàm \\(y = x^a\\) trên tập số dương và \\(0 \leq a \leq 1\\).

* Hàm logarithm \\(y = \log(x)\\) trên tập các số dương.

Dưới đây là đồ thị của một vài _concave functions_:
<hr>
<div class="imgcap">
 <img src ="/assets/16_convexity/concavefunctions.png" align = "center" width = "800">
 <div class = "thecap">Hình 9. Ví dụ về các concave functions một biến.</div>
</div>
<hr>

<a name="-affine-functions"></a>

#### 3.3.3. Affine functions
Các hàm số dạng \\(f(\mathbf{x}) = \mathbf{a}^T\mathbf{x} + b \\) vừa là convex, vừa là concave.

Khi biến là một ma trận \\(\mathbf{X}\\), các hàm affine được định nghĩa có dạng:
\\[
f(\mathbf{X}) = \text{trace}(\mathbf{A}^T\mathbf{X}) + b
\\]
trong đó \\(\text{trace}\\) là hàm số tính tổng các giá trị trên đường chéo của một ma trận vuông, \\(\mathbf{A}\\) là một ma trận có cùng chiều với \\(\mathbf{X}\\) (để đảm bảo phép nhân ma trận thực hiện được và kết quả là một ma trận vuông).

<a name="-quadratic-forms"></a>

#### 3.3.3. Quadratic forms
Hàm bậc hai một biến có dạng \\(f(x) = a x^2 + bx + c\\) là convex nếu \\(a > 0\\), là concave nếu \\(a < 0\\).

Với biến là một vector \\(\mathbf{x} = [x_1, x_2, \dots, x_n]\\), một quadratic form là một hàm số có dạng:
\\[
f(\mathbf{x}) = \mathbf{x}^T\mathbf{A}\mathbf{x} + \mathbf{b}^T\mathbf{x} + c
\\]
Với \\(\mathbf{A}\\) thường là một ma trận đối xứng, tức \\(a_{ij} = a_{ji}, \forall i, j\\), có số hàng bằng số phẩn tử của \\(\mathbf{x}\\), \\(\mathbf{b}\\) là một ma trận bất kỳ cùng chiều với \\(\mathbf{x}\\) và \\(c\\) là một hằng số bất kỳ.

Nếu \\(\mathbf{A}\\) là một ma trận (nửa) xác định dương thì \\(f(\mathbf{x})\\) là một _convex function_.

Nếu \\(\mathbf{A}\\) là một ma trận (nửa) xác định âm, tức \\(\mathbf{x}^T\mathbf{A}\mathbf{x} \leq 0, \forall \mathbf{x}\\), thì \\(f(\mathbf{x})\\) là một _concave function_.

_Các bạn có thể tìm đọc về ma trận xác định dương và các tính chất của nó trong sách Đại số tuyến tính bất kỳ. Nếu bạn gặp nhiều khó khăn trong phần này, hãy đọc lại kiến thức về Đại số tuyến tính, rất rất quan trọng trong Tối Ưu và Machine Learning._

[Hàm mất mát trong Linear Regression](/2016/12/28/linearregression/#ham-mat-mat) có dạng:
\\[
\begin{eqnarray}
\mathcal{L}(\mathbf{w}) &=& \frac{1}{2} \|\|\mathbf{y} - \mathbf{X}\mathbf{w}\|\|_2^2 = \frac{1}{2} (\mathbf{y} - \mathbf{X}\mathbf{w})^T(\mathbf{y} - \mathbf{X}\mathbf{w})  \\\
&=& \frac{1}{2} \mathbf{w}^T\mathbf{X}^T\mathbf{Xw} - \mathbf{y}^T\mathbf{Xw} + \frac{1}{2}\mathbf{y}^T\mathbf{y}
\end{eqnarray}
\\]
vì \\(\mathbf{X}^T\mathbf{X}\\) là một ma trận xác định dương, hàm mất mát của Linear Regression chính là một convex function.

<a name="-norms"></a>

#### 3.3.4. Norms
Vâng, lại là norms. Một hàm số bất kỳ thỏa mãn [ba điều kiện của norm](/math/#-norms-chuan) đều là một _convex function_. Bạn đọc có thể chứng minh điều này bằng định nghĩa.

Dưới đây là hai ví dụ về norm 1 (trái) và norm 2 (phải) với số chiều là 2 (chiều thứ ba trong hình dưới đây là giá trị của hàm số).
<hr>
<div>
<table width = "100%" style = "border: 0px solid white">
   <tr >
        <td width="40%" style = "border: 0px solid white">
        <img style="display:block;" width = "100%" src = "/assets/16_convexity/norm1_surf.png">
         </td>
        <td width="40%" style = "border: 0px solid white">
        <img style="display:block;" width = "100%" src = "/assets/16_convexity/norm2_surf.png">
        </td>
    </tr>
</table>
<div class = "thecap"> Hình 10: Ví dụ về mặt của các norm hai biến.
</div>
</div>
<hr>

Nhận thấy rằng các bề mặt này đều có _một đáy duy nhất_ tương ứng với gốc tọa độ (đây chính là điều kiện đầu tiên của norm). Các hàm _strictly convex_ khác cũng có dạng tương tự, tức có một _đáy_ duy nhất. Điều này cho thấy nếu ta _thả một hòn bi_ ở vị trí bất kỳ trên các bề mặt này, cuối cùng nó sễ _lăn_ về đáy. Nếu liên tưởng tới thuật toán [Gradient Descent](/2017/01/12/gradientdescent/) thì việc áp dụng thuật toán này vào các bài toán không ràng buộc với _hàm mục tiêu_ là _strictly convex_ (và giả sửa là khả vi, tức có đạo hàm) sẽ cho kết quả rất tốt nếu _learning rate_ không quá lớn. Đây chính là một trong các lý do vì sao các _convex functions_ là quan trọng, cũng là lý do vì sao tôi dành bài viết này chỉ để nói về _convexity_. (Bạn đọc được khuyến khích đọc hai bài về [Gradient Descent](/2017/01/12/gradientdescent/) trong blog này).

Tiện đây, tôi cũng lấy thêm hai ví dụ về các hàm không phải convex (cũng không phải concave). Hàm thứ nhất \\(f(x, y) = x^2 - y^2\\) là một hyperbolic, hàm thứ hai \\(f(x,y) = \frac{1}{10}(x^2 + 2y^2 - 2\sin(xy)) \\).




<hr>
<div>
<table width = "100%" style = "border: 0px solid white">
   <tr >
        <td width="40%" style = "border: 0px solid white">
        <img style="display:block;" width = "100%" src = "/assets/16_convexity/hyperbol.png">
         </td>
        <td width="40%" style = "border: 0px solid white">
        <img style="display:block;" width = "100%" src = "/assets/16_convexity/nonconvex_surface.png">
        </td>
    </tr>
</table>
<div class = "thecap">Hình 11: Ví dụ về các hàm hai biến không convex.
</div>
</div>
<hr>

**Contours - level sets**
Với các hàm số phức tạp hơn, khi vẽ các mặt trong không gian ba chiều sẽ khó tưởng tượng hơn, tức khó nhìn được tính _convexity_ của nó. Một phương pháp thường được sử dụng là dùng _contours_ hay _level sets_. Tôi cũng đã đề cập đến khái niệm này trong Bài Gradient Descent, phần [đường đồng mức](/2017/01/12/gradientdescent/#duong-dong-muc-level-sets).

Contours là cách mô tả các mặt trong không gian ba chiều bằng cách chiều nó xuống không gian hai chiều. Trong không gian hai chiều, các điểm thuộc cùng một _đường_ tương ứng với các điểm làm cho hàm số có giá trị bằng nhau. Mỗi _đường_ đó còn được gọi là một _level set_. Trong Hình 9 và Hình 10, các đường của các mặt lên mặt phẳng \\(0xy\\) chính là các _level sets_. Một cách hiểu khác, mỗi đường _level set_ là một _vết cắt_ nếu ta cắt các bề mặt bởi một mặt phẳng song song với mặt phẳng \\(0xy\\).

Khi thể hiện một hàm số hai biến để kiểm tra tính convexity của nó, hoặc để tìm điểm cực trị của nó, người ta thường vẽ _contours_ thay vì vẽ các mặt trong không gian ba chiều. Dưới đây là một vài ví dụ về contours:

<hr>
<div>
<table width = "100%" style = "border: 0px solid white">
   <tr >

        <td width="30%" style = "border: 0px solid white">
        <img style="display:block;" width = "100%" src = "/assets/16_convexity/abs_2d.png">
        </td>

        <td width="30%" style = "border: 0px solid white">
        <img style="display:block;" width = "100%" src = "/assets/16_convexity/norm_2d.png">
        </td>

        <td width="30%" style = "border: 0px solid white">
        <img style="display:block;" width = "100%" src = "/assets/16_convexity/max_2d.png">
         </td>
    </tr>

      <tr >
      <td width="30%" style = "border: 0px solid white">
      <img style="display:block;" width = "100%" src = "/assets/16_convexity/linear_2d.png">
       </td>

         <td width="30%" style = "border: 0px solid white">
         <img style="display:block;" width = "100%" src = "/assets/16_convexity/NE.png">
         </td>

         <td width="30%" style = "border: 0px solid white">
         <img style="display:block;" width = "100%" src = "/assets/16_convexity/hyper_2d.png">
         </td>
     </tr>
</table>
<div class = "thecap"> Hình 12: Ví dụ về Countours.
</div>
</div>
<hr>
Các đường màu càng xanh đậm thì tương ứng với các giá trị càng nhỏ, các đường màu càng đỏ đậm thì tương ứng các giá trị càng lớn.

Ở hàng trên, các đường _level sets_ là các đường khép kín (closed). Khi các đường kín này tập trung nhỏ dần ở một điểm thì các điểm đó là các điểm cực trị. Với các _convex functions_ như trong ba ví dụ này, chỉ có 1 điểm cực trị và đó cũng là điểm làm cho hàm số đạt giá trị nhỏ nhất (global optimal). Nếu để ý, bạn sẽ thấy các đường khép kín này tạo thành một _vùng lồi_!

Ở hàng dưới, các đường không phải khép kín. Hình bên trái tương ứng với một hàm tuyến tính \\(f(x, y) = x + y\\) và đó là một _convex function_. Hình ở giữa cũng là một _convex function_ (bạn có thể chứng minh điều này sau khi tính đạo hàm bậc hai, tôi sẽ nói ở phía dưới) nhưng các level sets là các _đường không kín_. Hàm này có \\(\log\\) nên tập xác định là góc phần tư thứ nhất tương ứng với các tọa độ dương (chú ý rằng tập hợp các điểm có tọa độ dương cũng là một _tập lồi_). Các _đường không kín_ này nếu kết hợp với trục \\(Ox, Oy\\) sẽ tạo thành biên của các _tập lồi_. Hình cuối cùng là contours của một hàm hyperbolic, hàm này không phải là _hàm lồi_.

<a name="-\\\alpha-\\-sublevel-sets"></a>

### 3.4. \\(\alpha-\\) sublevel sets
<hr>
**Định nghĩa:** \\(\alpha-\\)**sublevel set** của một hàm số \\(f : \mathbb{R}^n \rightarrow \mathbb{R}\\) được định nghĩa là:
\\[
\mathcal{C}_{\alpha} = \\{\mathbf{x} \in \text{dom} f ~\big\|~ f(\mathbf{x}) \leq \alpha \\}
\\]
<hr>
Tức tập hợp các điểm trong tập xác định của \\(f\\) mà tại đó, \\(f\\) đạt giá trị nhỏ hơn hoặc bằng \\(\alpha\\).


Quay lại với Hình 12, hàng trên, các \\(\alpha-\\) sublevel sets chính là phần bị bao bởi các level sets.

Ở hàng dưới, bên trái, các \\(\alpha-\\) sublevel sets chính là phần nửa mặt phẳng phía dưới xác định bởi các đường thẳng level sets. Ở hình giữa, các \\(\alpha-\\) sublevel sets chính là các vùng bị giới hạn bởi các trục tọa độ và các level sets.

Hàng dưới, bên phải, các \\(\alpha-\\) sublevel sets hơi khó tưởng tượng chút. Với \\(\alpha > 0\\), các level sets là các đường màu vàng hoặc đỏ. Các \\(\alpha-\\) sublevel sets tương ứng là phần _bị bóp vào trong_, giới hạn bởi các đường đỏ cùng màu. Các vùng này, có thể dễ nhận thấy, là _không lồi_.

<hr>
**Định lý:** Nếu một hàm số là lồi thì _mọi_ \\(\alpha-\\) sublevel sets của nó là lồi. Ngược lại chưa chắc đã đúng, tức nếu các \\(\alpha-\\) sublevel sets của một hàm số là _lồi_ thì hàm số đó chưa chắc đã _lồi_.
<hr>

Điều này chỉ ra rằng nếu tồn tại một giá trị \\(\alpha\\) sao cho một \\(\alpha-\\) sublevel set của một hàm số là _không lồi_, thì hàm số đó là _không lồi_ (không lồi nhưng không có nghĩa là _concave_, chú ý). Vậy nên Hyperbolic không phải là hàm lồi.

Các ví dụ ở hình 12, trừ hình cuối cùng, đều tương ứng với các hàm lồi.

Một ví dụ về việc một hàm số không _convex_ nhưng mọi \\(\alpha-\\) sublevel sets là _convex_ là hàm \\(f(x, y) = -e^{x+y}\\). Hàm này có mọi \\(\alpha-\\) sublevel sets là nửa mặt phẳng - là _convex_, nhưng nó không phải là _convex_ (trong trường hợp này nó là _concave_).

Dưới đây là một ví dụ khác về việc một hàm số có mọi \\(\alpha-\\) sublevel sets là _lồi_ nhưng không phải _hàm lồi_.
<hr>
<div class="imgcap">
 <img src ="/assets/16_convexity/sin_surf2.png" align = "center" width = "800">
 <div class = "thecap">Hình 13. Mọi alpha-sublevel sets là convex sets nhưng hàm số là nonconvex.</div>
</div>
<hr>

Mọi \\(\alpha-\\) sublevel sets của hàm số này đều là các hình tròn - _convex_ nhưng hàm số đó không phải là _lồi_. Vì có thể tìm được hai điểm trên mặt này sao cho đoạn thẳng nối hai điểm nằm hoàn toàn phía dưới của mặt (một điểm ở _cánh_ và 1 điểm ở _đáy_ chẳng hạn). 

Những hàm số có tập xác định là một _tập lồi_ và có mọi có \\(\alpha-\\) sublevel sets là _lồi_ được gọi chung là _quasiconvex_. Mọi _convex function_ đều là _quasiconvex_ nhưng ngược lại không đúng. Định nghĩa chính thức của _quasiconvex function_ được phát biểu như sau: 
<a name = "quasiconvex">
<hr>
**Quasiconvex function:**
Một hàm số \\(f: \mathcal{C} \rightarrow \mathbb{R}\\) với \\(\mathcal{C}\\) là một tập con _lồi_ của \\(\mathbb{R}^n\\) được gọi là _quasiconvex_ nếu với mọi \\(\mathbf{x}, \mathbf{y}) \in \mathcal{C}\\) và mọi \\(\theta \in [0, 1]\\), ta có: 
\\[
f(\theta\mathbf{x} + (1 - \theta)\mathbf{y}) \leq \max\\{f(\mathbf{x}), f(\mathbf{y})\\}
\\]
<hr> 
Định nghĩa này khác với định nghĩa về _convex function_ một chút. 

<a name="-kiem-tra-tinh-chat-loi-dua-vao-dao-ham"></a>

### 3.5. Kiểm tra tính chất lồi dựa vào đạo hàm.
Có một cách để nhận biết một hàm số khả vi có là hàm lồi hay không dựa vào các đạo hàm bậc nhất hoặc đạo hàm bậc hai của nó.
<a name="-first-order-condition"></a>

#### 3.5.1. First-order condition
Trước hết chúng ta định nghĩa phương trình đường (mặt) tiếp tuyến của một hàm số \\(f\\) khả vi tại một điểm nằm trên đồ thị (mặt) của hàm số đó \\((\mathbf{x}_0, f(\mathbf{x}_0)\\). Với hàm một biến, bạn đọc đã quen thuộc:
\\[
y = f'(x_0)(x - x_0) + f(x_0)
\\]
Với hàm nhiều biến, đặt \\(\nabla f(\mathbf{x}_0)\\) là gradient của hàm số \\(f\\) tại điểm \\(\mathbf{x}_0\\), phương trình mặt tiếp tuyến được cho bởi:
\\[
y = \nabla f(\mathbf{x}\_0)^T (\mathbf{x} - \mathbf{x}_0) + f(\mathbf{x}_0)
\\]
<hr>
**First-order condition** nói rằng: Giả sử hàm số \\(f\\) có tập xác định là một tập lồi, có đạo hàm tại mọi điểm trên tập xác định đó. Khi đó, hàm số \\(f\\) là _lồi_ **nếu và chỉ nếu** với mọi \\(\mathbf{x}, \mathbf{x}_0\\) trên tập xác định của hàm số đó, ta có:
\\[
f(\mathbf{x}) \geq f(\mathbf{x}_0) + \nabla f(\mathbf{x}\_0)^T(\mathbf{x} - \mathbf{x}_0) ~~ (6)
\\]
<hr>

Tương tự như thế, một hàm số là _stricly convex_ nếu dấu bằng trong \\((6)\\) xảy ra khi và chỉ khi \\(\mathbf{x} = \mathbf{x}_0\\).

Nói một cách trực quan hơn, một hàm số là lồi nếu đường (mặt) tiếp tuyến tại một điểm bất kỳ trên đồ thị (mặt) của hàm số đó **nằm dưới** đồ thị (mặt) đó.
(Đừng quên điều kiện về tập xác định là lồi)
Dưới đây là ví dụ về _hàm lồi_ và _hàm không lồi_.
<hr>
<div class="imgcap">
 <img src ="/assets/16_convexity/first_order.png" align = "center" width = "800">
 <div class = "thecap">Hình 13. Kiểm tra tính convexity dựa vào đạo hàm bậc nhất. Trái: hàm lồi, phải: hàm không lồi.</div>
</div>
<hr>
Hàm bên trái là một hàm lồi. Hàm bên phải không phải là hàm lồi vì đồ thị của nó vừa nằm trên, vừa nằm dưới tiếp tuyến.

(_iff_ là viết tắt của _if and only if_)

**Ví dụ:** Nếu ma trận đối xứng \\(\mathbf{A}\\) là _xác định dương_ thì hàm số \\(f(\mathbf{x}) = \mathbf{x}^T\mathbf{A}\mathbf{x}\\) là _hàm lồi_.

*Chứng minh:* Đạo hàm bậc nhất của hàm số trên là:

\\[
\nabla f(\mathbf{x}) = 2\mathbf{A} \mathbf{x}
\\]
Vậy _first-order condition_ có thể viết dưới dạng (chú ý rằng \\(\mathbf{A}\\) là một ma trận đối xứng):
\\[
\begin{eqnarray}
\mathbf{x}^T\mathbf{Ax} &\geq& 2(\mathbf{A}\mathbf{x}_0)^T (\mathbf{x} - \mathbf{x}_0) + \mathbf{x}_0^T\mathbf{A}\mathbf{x}_0 \\\
⇔ \mathbf{x}^T\mathbf{Ax} &\geq& 2\mathbf{x}_0^T\mathbf{A}\mathbf{x} -\mathbf{x}_0^T\mathbf{A}\mathbf{x}_0  \\\
⇔(\mathbf{x} - \mathbf{x}_0)^T\mathbf{A}(\mathbf{x} - \mathbf{x}_0) &\geq& 0
\end{eqnarray}
\\]

Bất đẳng thức cuối cùng là đúng dựa trên định nghĩa của một ma trận _xác định dương_. Vậy hàm số \\(f(\mathbf{x}) = \mathbf{x}^T\mathbf{A}\mathbf{x}\\) là _hàm lồi_.

_First-order condition_ ít được sử dụng để tìm tính chất lồi của một hàm số, thay vào đó, người ta thường dùng _Second-order condition_ với các hàm có đạo hàm tới bậc hai.
<a name="-second-order-condition"></a>

#### 3.5.2. Second-order condition
Với hàm nhiều biến, tức biến là một vector, giả sử có chiều là \\(d\\), đạo hàm bậc nhất của nó là một vector cũng có chiều là \\(d\\). Đạo hàm bậc hai của nó là một ma trận vuông có chiều là \\(d\times d\\). Đạo hàm bậc hai của hàm số \\(f(\mathbf{x})\\) được ký hiệu là \\(\nabla^2 f(\mathbf{x})\\). Đạo hàm bậc hai còn được gọi là _Hessian_.

<hr>
**Second-order condition:** Một hàm số có đạo hàm bậc hai là _convex_ nếu **dom**\\(f\\) là _convex_ và Hessian của nó là một ma trận _nửa xác định dương_ với mọi \\(\mathbf{x}\\) trong tập xác định:
\\[
\nabla^2 f(\mathbf{x}) \succeq 0.
\\]
<hr>
Nếu Hessian là một ma trận _xác định dương_ thì hàm số đó _strictly convex_.
Tương tự, nếu Hessian là một ma trận _xác định âm_ thì hàm số đó là _strictly concave_.

Với hàm số một biến \\(f(x)\\), điều kiện này tương đương với \\(f"(x) \geq 0\\) với mọi \\(x\\) thuộc tập xác định (và tập xác định là _lồi_).

**Ví dụ:**

* Hàm _negative entropy_ \\(f(x) = x\log(x)\\) là _stricly convex_ vì tập xác định là \\(x > 0\\) là một tập lồi và \\(f"(x) = 1/x\\) là một số dương với mọi \\(x\\) thuộc tập xác định.

* Hàm \\(f(x) = x^2 + 5\sin(x)\\) không là hàm lồi vì đạo hàm bậc hai \\(f"(x) = 2 - 5\sin(x)\\) có thể nhận giá trị âm.

* Hàm _cross entropy_ là một hàm _strictly convex_. Xét ví dụ đơn giản với chỉ hai xác suất \\(x\\) và \\(1 - x\\) với \\(a\\) là một hằng số thuộc đoạn \\([0, 1]\\) và \\(0 < x < 1\\): \\(f(x) = -(a \log(x) + (1 - a) \log(1 - x))\\) có đạo hàm bậc hai là \\(\frac{a}{x^2} + \frac{1 - a}{(1-x)^2}\\) là một số dương.

* Nếu \\(\mathbf{A}\\) là một ma trận xác định dương thì \\(f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^T\mathbf{Ax}\\) là lồi vì Hessian của nó chính là \\(\mathbf{A}\\) là một ma trận xác định dương.

* Xét hàm số _negative entropy_ với hai biến: \\(f(x, y) = x \log(x) + y \log(y)
\\) trên tập các giá trị dương của \\(x\\) và \\(y\\). Hàm số này có đạo hàm bậc nhất là \\([\log(x) + 1, \log(y) + 1]^T\\) và Hessian là:
\\[
\left[
\begin{matrix}
1/x & 0 \\\
0 & 1/y
\end{matrix}
\right]
\\]
là một ma trận đường chéo với các thành phần trên đường chéo là dương nên là một ma trận xác định dương. Vậy _negative entropy_ là một hàm _strictly convex_.(_Chú ý rằng một ma trận là xác định dương nếu các trị riêng của nó đều dương. Với một ma trận là ma trận đường chéo thì các trị riêng của nó chính là các thành phần trên đường chéo_.)




Ngoài ra còn nhiều tính chất thú vị của các _hàm lồi_, các bạn được khuyến khích đọc thêm Chương 3 của cuốn Convex Optimization trong phần tài liệu tham khảo.

<a name="-tom-tat"></a>

## 4. Tóm tắt

* Machine Learning và Optimization có quan hệ mật thiết với nhau. Trong Optimization, Convex Optimization là quan trọng nhất. Một bài toán là convex optimization nếu _hàm mục tiêu_ là convex và tập hợp các điểm thỏa mãn các điều kiện ràng buộc là một _convex set_.

* Trong _convex set_, mọi đoạn thẳng nối hai điểm bất kỳ trong tập đó sẽ nằm hoàn toàn trong tập đó. Tập hợp các giao điểm của các _convex sets_ là một _convex set_.

* Một hàm số là _convex_ nếu đoạn thẳng nối hai điểm bất kỳ trên đồ thì hàm số đó không nằm dưới đồ thị đó.

* Một hàm số khả vi là _convex_ nếu tập xác định của nó là _convex_ và đường (mặt) tiếp tuyến _không nằm phía trên_ đồ thị (bề mặt) của hàm số đó.

* Các norms là các hàm lồi, được sử dụng nhiều trong tối ưu.

<a name="-tai-lieu-tham-khao"></a>

## 5. Tài liệu tham khảo

[1] [Convex Optimization](http://stanford.edu/~boyd/cvxbook/) – Boyd and Vandenberghe, Cambridge University Press, 2004.


