---
layout: post
title: Math
permalink: /math/
mathjax: true
---

Một số kiến thức về Đại Số Tuyến Tính, Xác Suất Thống Kê, Toán Tối Ưu cần thiết cho Machine Learning.

(_đang trong thời gian xây dựng, cập nhật theo bài_)

**Trong trang này:**
<!-- MarkdownTOC -->

- [Đại số tuyến tính](#dai-so-tuyen-tinh)
    - [Chuẩn \(Norms\)](#chuan-norms)
    - [Bảng các đạo hàm cơ bản](#bang-cac-dao-ham-co-ban)
        - [Cho vector](#cho-vector)
        - [Cho ma trận](#cho-ma-tran)

<!-- /MarkdownTOC -->

**Lưu ý về ký hiệu toán học:** _trong các bài viết của tôi, các số vô hướng được biểu diễn bởi các chữ cái viết ở dạng không in đậm, có thể viết hoa, ví dụ \\(x_1, N, y, k\\). Các vector được biểu diễn bằng các chữ cái thường in đậm, ví dụ \\(\mathbf{y}, \mathbf{x}_1 \\). Nếu không giải thích gì thêm, các vector được mặc định hiểu là các vector cột. Các ma trận được biểu diễn bởi các chữ viết hoa in đậm, ví dụ \\(\mathbf{X, Y, W} \\)._

<!-- ========================== New Heading ==================== -->
<a name="dai-so-tuyen-tinh"></a>

## Đại số tuyến tính 

<!-- ========================== New Heading ==================== -->
<a name="chuan-norms"></a>

### Chuẩn (Norms)

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