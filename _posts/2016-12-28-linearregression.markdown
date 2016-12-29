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

- [Phân tích toán học](#phân-tích-toán-học)
- [Linear Regression trong Python](#python)
- [Thảo luận](#thảo-luận)
  - [Hạn chế của Linear Regression](#limitations)
  - [Các phương pháp tối ưu](#optimization)
- [Linear Regression](#linear-regression)

<!-- /MarkdownTOC -->

## Phân tích toán học 

Quay lại [ví dụ đơn giản được nêu trong bài trước](/2016/12/27/categories/): một căn nhà rộng \\(x_1 ~ \text{m}^2\\), có \\(x_2\\) phòng ngủ và cách trung tâm thành phố \\(x_3~ \text{km}\\) có giá là bao nhiêu. Giả sử chúng ta đã có số liệu thống kê từ 1000 căn nhà trong thành phố đó, liệu rằng khi có một căn nhà mới với các thông số về diện tích, số phòng ngủ và khoảng cách tới trung tâm, chúng ta có thể dự đoán được giá của căn phòng đó không? Nếu có thì hàm dự đoán \\(y = f(\mathbf{x}) \\) sẽ có dạng như thế nào. Ở đây \\(\mathbf{x} = [x_1, x_2, x_3] \\) là một vector hàng chưa thông tin _input_, \\(y\\) là một số vô hướng (scalar) biểu diễn _output_ (tức giá của căn nhà trong ví dụ này) 

**Lưu ý về ký hiệu toán học:** _trong các bài viết của tôi, các số vô hướng được biểu diễn bởi các chữ cái viết ở dạng không in đậm, có thể viết hoa, ví dụ \\(x_1, N, y, k\\). Các vector được biểu diễn bằng các chữ cái thường in đậm, ví dụ \\(\mathbf{y}, \mathbf{x}_1 \\). Các ma trận được biểu diễn bởi các chữ viết hoa in đậm, ví dụ \\(\mathbf{X, Y, W} \\)._


## Linear Regression trong Python[python]

## Thảo luận
 
\\(f(\mathbf{x})\\) là một đa thức bậc cao. 

### Hạn chế của Linear Regression[limitations]
* Nhạy cảm với nhiễu 
* Không biểu diễn được các mô hình phức tạp 

### Các phương pháp tối ưu[optimization]



<!-- Giả sử chúng ta có các cặp (_input, outcome_) \\( (\mathbf{x}_1, \mathbf{y}_1), \dots, (\mathbf{x}_N, \mathbf{y}_N) \\), chúng ta phải tìm một hàm  -->
## Linear Regression 

[Data](http://people.sc.fsu.edu/~jburkardt/datasets/regression/regression.html)