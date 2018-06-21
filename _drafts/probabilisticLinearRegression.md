## Ví dụ về Bayseian linear regression

Trong ví dụ này, chúng ta giả sử bias của mô hình đã được tích hợp vào vector hệ số thông qua [bias trick](https://machinelearningcoban.com/2017/04/28/multiclasssmv/#-bias-trick).

Bài toán đặt ra là: có các cặp dữ liệu (input, output) cho training \\(\\{(\mathbf{x}\_1, y_1, \dots, \mathbf{x}\_N, y_N)\\}\\). Hãy xác định một mô hình Linear Regression sao cho đầu ra là một biến ngẫu nhiên tuân theo phân phối chuẩn chứ không phải là một giá trị xác định.

Việc tìm biến đầu ra tuân theo một phân phối nào đó, ở đây là phân phối chuẩn, khiến cho mô hình một phần nào đó _linh hoạt_ hơn.

### Linear Regression
Việc đầu tiên ta cần làm là đi tìm **Model** phù hợp cho bài toán. Vì ta muốn đầu ra là một biến ngâu nhiên tuân theo phân phối chuẩn, ta có thể chọn mô hình:

\\[
  p(y \| \mathbf{x}, \theta) = \text{Norm}_{y}[\mathbf{w}^T\mathbf{x}, \sigma^2]
\\]

Phát biểu bằng lời: Nếu biết input \\(\mathbf{x}\\), vector hệ số \\(\mathbf{w}\\) và phương sai của phân phối chuẩn \\(\sigma^2\\), đầu ra là một giá trị tuân theo phân phối chuẩn có kỳ vọng là \\(\mathbf{w}^T\mathbf{x}\\) và phương sai là \\(\sigma^2\\).

Ta gọi mô hình này là _Probabilistic Linear Regression_.

Tham số mô hình ở đây là vector hệ số \\(\mathbf{w}\\) và phương sai \\(\sigma^2\\). Các tham số này có thể được giải bằng **Learning**. Ở đây, chúng ta sẽ so sánh 2 cách đánh giá các tham số này, bằng việc sử dụng MLE và MAP.

Việc **Inference** là tương đối đơn giản, bạn có thể đánh giá output nằm trong một khoảng nào đó với một mức độ _confidence_ nào đó, hoặc tìm một giá trị của output tại điểm mà hàm mật độ xác suất là lớn nhất, ở đây chính là kỳ vọng của phân phối chuẩn, và bằng \\(\mathbf{w}^T\mathbf{x}\\).

Khác với [Linear Regression thông thường](https://machinelearningcoban.com/2016/12/28/linearregression/), cách xây dựng mô hình Probabilistic Linear Regression này còn giúp ta tính toán được phương sai, tức cho phép ta biết độ _confidence_ của mô hình. Các bạn sẽ thấy việc này sau khi chúng ta tìm ra kết quả và quan sát hình vẽ mô tả nghiệm.

### Maximum Likelihood Estimation

Các tham số \\(\mathbf{w}\\) và \\(\sigma^2\\) có thể được tìm bằng MLE, với giả sử rằng các cặp (input, output) là độc lập:
\\[
\begin{eqnarray}
  (\mathbf{w}, \sigma) & = & \arg\max_{\mathbf{w}, \sigma} \left\[ \prod_{i=1}^N \frac{1}{\sqrt{2\pi \sigma^2}}\exp(- \frac{(y_i - \mathbf{w}^T\mathbf{x})}{2\sigma^2})\right\] \\\
  & = & \arg\max_{\mathbf{w}, \sigma} \left\[ \frac{1}{\sigma^N} \exp\left\( - \frac{\sum_{i=1}^N (y_i - \mathbf{w}^T\mathbf{x}\_i)}{2\sigma^2}\right\)\right\] \\\
  & = & \arg\max_{\mathbf{w}, \sigma} \left\[ - N \log \sigma - \frac{ \|\|\mathbf{y} - \mathbf{w}^T\mathbf{X}\|\|_F^2}{2\sigma^2}\right\] & (41)
\end{eqnarray}
\\]

với \\(\mathbf{y} = [y_1, \dots, y_N]^T\\) là 1 vector cột chứa các training output và \\(\mathbf{X} = [\mathbf{x}\_1, \dots, \mathbf{x}\_N]\\) là ma trận dữ liệu.

Từ đây ta có thể thấy rằng \\(\mathbf{w}\\) phải là nghiệm của bài toán:
\\[
  \mathbf{w} = \arg\min_{\mathbf{w}} \|\|\mathbf{y} - \mathbf{X}^T\mathbf{w}\|\|_F^2
\\]
Đây chính là bài toán Linear Regression quen thuộc với nghiệm:
\\[
  \mathbf{w} = (\mathbf{X}\mathbf{X}^T)^{-1} \mathbf{Xy} ~~~ (42)
\\]

Giải phương trình đạo hàm theo \\(\sigma^2\\) của biểu thức trong dầu ngoặc vuông của \\((41)\\) ta sẽ có:

\\[
  \sigma^2 = \frac{\|\|\mathbf{y} - \mathbf{X}^T\mathbf{w}\|\|_F^2}{N} ~~~(43)
\\]

Đây chính là trung bình của bình phương của lỗi. Như vậy, với Probabilistic Linear Regression, ta có thêm một tham số đánh giá sai số mô hình. Sai số càng nhỏ càng chứng tỏ rằng mô hình ta chọn càng chính xác. 

### Maximum A Posterior
