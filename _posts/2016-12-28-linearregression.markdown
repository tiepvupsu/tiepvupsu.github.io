---
layout: post
comments: true
title:  "Linear Regression"
date:   2016-12-28 15:22:00
mathjax: true
---

## Bài viết đang trong giai đoạn xây dựng, sẽ được hoàn thành sớm. 

<div class="imgcap">
<div >
<a href = "/2016/12/28/linearregression/">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Linear_regression.svg/400px-Linear_regression.svg.png" width = "500"></a>
    <!-- <img src="/assets/rl/mdp.png" height="206"> -->
</div>
<div class="thecap"> Linear Regression <br> (Nguồn: <a href ="https://en.wikipedia.org/wiki/Linear_regression">Wikipedia</a>)</div>
</div>

Trong bài này, tôi sẽ giới thiều một trong những thuật toán cơ bản nhất (và đơn giản nhất) của Machine Learning. Đây là một thuật toán _Supervised learning_ có tên **Linear Regression** (Hồi Quy Tuyến Tính). 

Trong trang này:
<!-- MarkdownTOC autolink="true" bracket="round" depth="0" style="unordered" indent="  " autoanchor="false" -->

- [1. Giới thiệu](#1-giới-thiệu)
- [2. Phân tích toán học](#2-phân-tích-toán-học)
- [3. Triển khai trên trên Python](#3-triển-khai-trên-trên-python)
- [4. Thảo luận](#4-thảo-luận)
  - [Output là một vector nhiều biến](#output-là-một-vector-nhiều-biến)
  - [Mô hình là một đa thức bậc cao](#mô-hình-là-một-đa-thức-bậc-cao)
  - [Hạn chế của Linear Regression](#hạn-chế-của-linear-regression)
  - [Các phương pháp tối ưu](#các-phương-pháp-tối-ưu)
- [5. Tài liệu tham khảo](#5-tài-liệu-tham-khảo)

<!-- /MarkdownTOC -->

## 1. Giới thiệu

Quay lại [ví dụ đơn giản được nêu trong bài trước](/2016/12/27/categories/): một căn nhà rộng \\(x_1 ~ \text{m}^2\\), có \\(x_2\\) phòng ngủ và cách trung tâm thành phố \\(x_3~ \text{km}\\) có giá là bao nhiêu. Giả sử chúng ta đã có số liệu thống kê từ 1000 căn nhà trong thành phố đó, liệu rằng khi có một căn nhà mới với các thông số về diện tích, số phòng ngủ và khoảng cách tới trung tâm, chúng ta có thể dự đoán được giá của căn phòng đó không? Nếu có thì hàm dự đoán \\(y = f(\mathbf{x}) \\) sẽ có dạng như thế nào. Ở đây \\(\mathbf{x} = [x_1, x_2, x_3] \\) là một vector hàng chưa thông tin _input_, \\(y\\) là một số vô hướng (scalar) biểu diễn _output_ (tức giá của căn nhà trong ví dụ này).

**Lưu ý về ký hiệu toán học:** _trong các bài viết của tôi, các số vô hướng được biểu diễn bởi các chữ cái viết ở dạng không in đậm, có thể viết hoa, ví dụ \\(x_1, N, y, k\\). Các vector được biểu diễn bằng các chữ cái thường in đậm, ví dụ \\(\mathbf{y}, \mathbf{x}_1 \\). Các ma trận được biểu diễn bởi các chữ viết hoa in đậm, ví dụ \\(\mathbf{X, Y, W} \\)._

Một cách đơn giản nhất, chúng ta có thể thấy rằng: i) diện tích nhà càng lớn thì giá nhà càng cao; ii) số lượng phòng ngủ càng lớn thì giá nhà càng cao; iii) càng xa trung tâm thì giá nhà càng giảm. Một hàm số đơn giản nhất có thể mô tả mối quan hệ giữa giá nhà và 3 đại lượng đầu vào là: 

\\[ \text{ giá nhà } = w_1 (\text{diện tích}) + w_2 (\text{số phòng}) + w_3 (\text{ khoảng cách}) + w_0 \\] 

trong đó, \\(w_1, w_2\\) là các hằng số dương, \\(w_3\\) là một hằng số âm, \\(w_0\\) là một hằng số được gọi là bias. Hay nói cách khác: 

\\[y = w_0 + w_1 x_1 + w_2 x_2 + w_3 x_3 = f(\mathbf{x})\\]

Mối quan hệ \\(y = f(\mathbf{x})\\) bên trên là một mối quan hệ tuyến tính (linear). Bài toán chúng ta đang làm là một bài toán thuộc loại regression. Bài toán đi tìm các hệ số tối ưu \\( (w_0, w_1, w_2, w_3)\\) chính vì vậy được gọi là bài toán Linear Regression. 

**Chú ý:** _Linear_ hay _tuyến tính_ hiểu một cách đơn giản là _thẳng, phẳng_. Trong không gian hai chiều, một hàm số được gọi là _tuyến tính_ nếu đồ thị của nó có dạng một _đường thẳng_. Trong không gian ba chiều, một hàm số được goi là _tuyến tính_ nếu đồ thị của nó có dạng một _mặt phẳng_. Trong không gian nhiều hơn 3 chiều, khái niệm _mặt phẳng_ không còn phù hợp nữa, thay vào đó, một khái niệm khác ra đời được gọi là _siêu mặt phẳng_ (_hyperplane_). Các hàm số tuyến tính là các hàm đơn giản nhất, vì chúng thuận tiện trong việc hình dung và tính toán. Chúng ta sẽ được thấy trong các bài viết sau, _tuyến tính_ rất quan trọng và hữu ích trong các bài toán Machine Learning. Kinh nghiệm cá nhân tôi cho thấy, trước khi hiểu được các thuật toán _phi tuyến_ (non-linear, không phẳng), chúng ta cần nắm vững các kỹ thuật cho các mô hình _tuyến tính_.


## 2. Phân tích toán học 



## 3. Triển khai trên trên Python

## 4. Thảo luận

### Output là một vector nhiều biến

\\(f(\mathbf{x})\\) là một đa thức bậc cao. 

### Mô hình là một đa thức bậc cao

### Hạn chế của Linear Regression
* Nhạy cảm với nhiễu 
* Không biểu diễn được các mô hình phức tạp 

### Các phương pháp tối ưu



<!-- Giả sử chúng ta có các cặp (_input, outcome_) \\( (\mathbf{x}_1, \mathbf{y}_1), \dots, (\mathbf{x}_N, \mathbf{y}_N) \\), chúng ta phải tìm một hàm  -->
## 5. Tài liệu tham khảo

[Data](http://people.sc.fsu.edu/~jburkardt/datasets/regression/regression.html)