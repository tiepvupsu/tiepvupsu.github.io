---
layout: post
title: Math
permalink: /math/
mathjax: true
<!-- tags: General -->
---

Một số kiến thức về Đại Số Tuyến Tính, Xác Suất Thống Kê, Toán Tối Ưu cần thiết cho Machine Learning.

(_đang trong thời gian xây dựng, cập nhật theo bài_)

**Trong trang này:**
<!-- MarkdownTOC -->

- [Lưu ý về ký hiệu](#luu-y-ve-ky-hieu)
- [Đại số tuyến tính](#dai-so-tuyen-tinh)
    - [Norms \(chuẩn\)](#norms-chuan)
        - [Định nghĩa](#dinh-nghia)
        - [Một số chuẩn thường dùng](#mot-so-chuan-thuong-dung)
        - [Chuẩn của ma trận](#chuan-cua-ma-tran)
    - [Bảng các đạo hàm cơ bản](#bang-cac-dao-ham-co-ban)
        - [Cho vector](#cho-vector)
        - [Cho ma trận](#cho-ma-tran)

<!-- /MarkdownTOC -->

<!-- ========================== New Heading ==================== -->
<a name="luu-y-ve-ky-hieu"></a>

## Lưu ý về ký hiệu

Trong các bài viết của tôi, các số vô hướng được biểu diễn bởi các chữ cái viết ở dạng không in đậm, có thể viết hoa, ví dụ \\(x_1, N, y, k\\). Các vector được biểu diễn bằng các chữ cái thường in đậm, ví dụ \\(\mathbf{y}, \mathbf{x}_1 \\). Nếu không giải thích gì thêm, các vector được mặc định hiểu là các vector cột. Các ma trận được biểu diễn bởi các chữ viết hoa in đậm, ví dụ \\(\mathbf{X, Y, W} \\).

Đối với vector, \\(\mathbf{x} = [x_1, x_2, \dots, x_n]\\) được hiểu là một vector hàng. Trong khi \\(\mathbf{x} = [x_1; x_2; \dots; x_n] \\) được hiểu là vector cột. Chú ý sự khác nhau giữa dầu phẩy (\\(,\\)) và dấu chấm phẩy (\\(;\\)). Đây chính là ký hiệu mà Matlab hay dùng.

Tương tự, trong ma trận, \\(\mathbf{X} = [\mathbf{x}\_1, \mathbf{x}\_2, \dots, \mathbf{x}_n]\\) được hiểu là các vector \\(\mathbf{x}\_j\\) được đặt cạnh nhau theo thứ tự từ trái qua phải để tạo ra ma trận \\(\mathbf{X}\\). Trong khi \\(\mathbf{X} = [\mathbf{x}\_1; \mathbf{x}_2; \dots; \mathbf{x}_m]\\) được hiểu là các vector \\(\mathbf{x}_i\\) được đặt chồng lên nhau theo thứ tự từ trên xuống dưới dể tạo ra ma trận \\(\mathbf{X}\\). Các vector được ngầm hiểu là có kích thước phù hợp để có thể xếp cạnh hoặc xếp chồng lên nhau.

<!-- ========================== New Heading ==================== -->
<a name="dai-so-tuyen-tinh"></a>

## Đại số tuyến tính 

<!-- ========================== New Heading ==================== -->
<a name="norms-chuan"></a>

### Norms (chuẩn)
Trong không gian một chiều, việc đo khoảng cách giữa hai điểm đã rất quen thuộc: lấy trị tuyệt đối của hiệu giữa hai giá trị đó. Trong không gian hai chiều, tức mặt phẳng, chúng ta thường dùng khoảng cách Eclid để đo khoảng cách giữa hai điểm. Khoảng cách này chính là cái chúng ta thường nói bằng ngôn ngữ thông thường là _đường chim bay_. Đôi khi, để đi từ một điểm này tới một điểm kia, con người chúng ta không thể đi bằng đường chim bay được mà còn phụ thuộc vào việc đường đi nối giữa hai điểm có dạng như thế nào nữa. 

Việc đo khoảng cách giữa hai điểm dữ liệu nhiều chiều, tức hai vector, là rất cần thiết trong Machine Learning. Chúng ta cần đánh giá xem điểm nào là điểm gần nhất của một điểm khác; chúng ta cũng cần đánh giá xem độ chính xác của việc ước lượng; và trong rất nhiều ví dụ khác nữa. 

Và đó chính là lý do mà khái niệm norm (chuẩn) ra đời. Có nhiều loại norm khác nhau mà các bạn sẽ thấy ở dưới đây: 

Để xác định khoảng cách giữa hai vector \\(\mathbf{y}\\) và \\(\mathbf{z}\\), người ta thường áp dụng một hàm số lên vector hiệu \\(\mathbf{x = y - z}\\). Một hàm số được dùng để đo các vector cần có một vài tính chất đặc biệt. 
<!-- ========================== New Heading ==================== -->
<a name="dinh-nghia"></a>

#### Định nghĩa
Một hàm số \\(f() \\) ánh xạ một điểm \\(\mathbf{x}\\) từ không gian \\(n\\) chiều sang tập số thực một chiều được gọi là norm nếu nó thỏa mãn ba điều kiện sau đây:

1. \\(f(\mathbf{x}) \geq 0\\). Dấu bằng xảy ra \\(\Leftrightarrow \mathbf{x = 0} \\).
2. \\(f(\alpha \mathbf{x}) = \|\alpha\| f(\mathbf{x}), ~~~\forall \alpha \in \mathbb{R}\ \\)
3. \\(f(\mathbf{x}_1) + f(\mathbf{x}_2) \geq f(\mathbf{x}_1 + \mathbf{x}_2), ~~\forall \mathbf{x}_1, \mathbf{x}_2 \in \mathbf{R}^n\\)

**Điều kiện thứ nhất** là dễ hiểu vì khoảng cách không thể là một số âm. Hơn nữa, khoảng cách giữa hai điểm \\(\mathbf{y}\\) và \\(\mathbf{z}\\) bằng 0 nếu và chỉ nếu hai điểm nó trùng nhau, tức \\(\mathbf{x = y - z = 0} \\).

**Điều kiện thứ hai** cũng có thể được lý giải như sau. Nếu ba điểm \\(\mathbf{y, v}\\) và \\(\mathbf{z}\\) thẳng hàng, hơn nữa \\(\mathbf{v - y} = \alpha (\mathbf{v - z}) \\) thì khoảng cách giữa \\(\mathbf{v}\\) và \\(\mathbf{y}\\) sẽ gấp \\( \|\alpha \|\\) lần khoảng cách giữa \\(\mathbf{v}\\) và \\(\mathbf{z}\\).

**Điều kiện thứ ba** chính là bất đẳng thức tam giác nếu ta coi \\(\mathbf{x}_1 = \mathbf{ w - y}, \mathbf{x}_2 = \mathbf{z - w} \\) với \\(\mathbf{w}\\) là một điểm bất kỳ trong cùng không gian.


<!-- ========================== New Heading ==================== -->
<a name="mot-so-chuan-thuong-dung"></a>

#### Một số chuẩn thường dùng

Giả sử các vectors \\(\mathbf{x} = [x_1; x_2; \dots; x_n]\\), \\(\mathbf{y} = [y_1; y_2; \dots; y_n]\\).

<a name = "norm2"></a>
Nhận thấy rằng khoảng cách Euclid chính là một norm, norm này thường được gọi là **norm 2**:
\\[
\|\|\mathbf{x}\|\|\_2 = \sqrt{x_1^2 + x_2^2 + \dots x_n^2}
\\]

<a name = "normp"></a>

Với \\(p\\) __là một số không nhỏ hơn 1__ bất kỳ, hàm số sau đây:
\\[
\|\|\mathbf{x}\|\|\_p = (\|x_1\|^p + \|x_2\|^p + \dots \|x_n\|^p)^{\frac{1}{p}} ~~(1)
\\]

được chứng minh thỏa mãn ba điều kiện bên trên, và được gọi là **norm p**. 

<a name = "norm0"></a>

Nhận thấy rằng khi \\(p \rightarrow 0 \\) thì biểu thức bên trên trở thành _số các phần tử khác 0 của_ \\(\mathbf{x}\\). Hàm số  \\((1)\\) khi \\(p = 0\\) được gọi là giả chuẩn (pseudo-norm) 0. Nó không phải là norm vì nó không thỏa mãn điều kiện 2 và 3 của norm. Giả-chuẩn này, thường được ký hiệu là \\(\|\|\mathbf{x}\|\|_0\\), khá quan trọng trong Machine Learning vì trong nhiều bài toán, chúng ta cần có ràng buộc “sparse”, tức số lượng thành phần “active” của \\(\mathbf{x}\\) là nhỏ. 

Có một vài giá trị của \\(p\\) thường được dùng:

1. Khi \\(p = 2\\) chúng ta có norm 2 như ở trên.

<a name = "norm1"></a>

2. Khi \\(p = 1\\) chúng ta có:
\\[
\|\|\mathbf{x}\|\|_1 = \|x_1\| + \|x_2\| + \dots \|x_n\|
\\]
là tổng các trị tuyệt đối của từng phần tử của \\(\mathbf{x}\\). Norm 1 thường được dùng như xấp xỉ của norm 0 trong các bài toán có ràng buộc "sparse". Dưới đây là một ví dụ so sánh norm 1 và norm 2 trong không gian hai chiều:
<div class="imgcap">
<img src ="/assets/norm12.png" width = "500" align = "center">
<div class="thecap"> Norm 1 và norm 2 trong không gian hai chiều.<br></div>
</div> 
Norm 2 (màu xanh) chính là đường thằng "chim bay" nối giữa hai vector \\(\mathbf{x} \\) và \\(\mathbf{y}\\). Khoảng cách norm 1 giữa hai điểm này (màu đỏ) có thể diễn giải như là đường đi từ \\(\mathbf{x} \\) tới \\(\mathbf{y}\\) trong một thành phố mà đường phố tạo thành hình bàn cờ. Chúng ta chỉ có cách đi dọc theo cạnh của bàn cờ mà không được đi thẳng.

3. Khi \\(p \rightarrow \infty \\), ta có norm \\(p\\) chính là trị tuyệt đối của phần tử lớn nhất của vector đó:
\\[
\|\|\mathbf{x}\|\|\_{\infty} = \max_{i = 1, 2, \dots, n} \|x\_i\|
\\]

<!-- ========================== New Heading ==================== -->
<a name="chuan-cua-ma-tran"></a>

#### Chuẩn của ma trận
Với một ma trận \\(\mathbf{A} \in \mathbb{R}^{m\times n}\\), chuẩn thường được dùng nhất là chuẩn Frobenius, ký hiệu là \\(\|\|\mathbf{A}\|\|\_F\\) là căn bậc hai của tổng bình phương tất cả các phần tử của ma trận đó. 
\\[
\|\|\mathbf{A}\|\|\_F = \sqrt{\sum_{i = 1}^m \sum_{j = 1}^n a\_{ij}^2}
\\]

<!-- ========================== New Heading ==================== -->
<a name="bang-cac-dao-ham-co-ban"></a>

### Bảng các đạo hàm cơ bản 
(_Đừng sợ, chỉ cần dùng để tra cứu thôi_)
<!-- ========================== New Heading ==================== -->
<a name="cho-vector"></a>

#### Cho vector 

| \\(f(\mathbf{x}) \\)           | \\( \frac{\partial f(\mathbf{x})}{\partial \mathbf{x}} \\)     |         
| :----------------------:       | :------------------------------------------------------------: |         
| \\(\mathbf{a}^T \mathbf{x} \\) | \\(\mathbf{a}\\)                                               |         
| \\(\mathbf{x}^T \mathbf{x} =  \\| \\|\mathbf{x} \\|\\|_2^2 \\)  | \\(2\mathbf{x}  \\)     |
| \\( \|\|\mathbf{Ax-b} \|\|_2^2 \\)  | \\( 2\mathbf{A}^T (\mathbf{Ax - b})\\)      |         
| \\(\mathbf{a}^T\mathbf{x}^T\mathbf{xb} \\) |  \\(2\mathbf{a}^T\mathbf{bx} \\) |
| \\(\mathbf{a}^T\mathbf{x}\mathbf{x}^T\mathbf{b} \\) |  \\( (\mathbf{a}\mathbf{b}^T + \mathbf{b}\mathbf{a}^T) \mathbf{x} \\) |

<!-- ========================== New Heading ==================== -->
<a name="cho-ma-tran"></a>

#### Cho ma trận

| \\(f(\mathbf{x}) \\)           | \\( \frac{\partial f(\mathbf{x})}{\partial \mathbf{x}} \\)     |       
| :----------------------:       | :------------------------------------------------------------: |         
| \\( \mathbf{a}^T \mathbf{X}^T \mathbf{Xb}\\) | \\( \mathbf{X}(\mathbf{ab}^T + \mathbf{ba}^T)        \\) |
| \\( \mathbf{a}^T \mathbf{X} \mathbf{X}^T \mathbf{b}\\) | \\( (\mathbf{ab}^T + \mathbf{ba}^T)\mathbf{X}        \\) |
| \\( \mathbf{a}^T \mathbf{Y} \mathbf{X}^T \mathbf{b}\\) | \\( \mathbf{b}\mathbf{a}^T \mathbf{Y}        \\) |
| \\( \mathbf{a}^T \mathbf{Y}^T \mathbf{X} \mathbf{b}\\) | \\( \mathbf{Y}\mathbf{a}\mathbf{b}^T         \\) |
| \\( \mathbf{a}^T \mathbf{X} \mathbf{Y}^T \mathbf{b}\\) | \\( \mathbf{a}\mathbf{b}^T\mathbf{Y}         \\) |
| \\( \mathbf{a}^T \mathbf{X}^T \mathbf{Y} \mathbf{b}\\) | \\( \mathbf{Y}\mathbf{b}\mathbf{a}^T         \\) |