---
layout: post
comments: true
title:  "Bài 17: Convex Optimization Problems"
title2:  "17. Convex Optimization Problems"
date:   2017-03-19 15:22:00
permalink: 2017/03/19/convexopt/
mathjax: true
tags: Convex Optimization
category: Optimization
sc_project: 11288928
sc_security: 0edaf8bc
img: /assets/17_convexopt/optimalitycondition.png
summary: Giới thiệu về các bài toán Convex Optimization
---
<!-- MarkdownTOC depth = 3 -->

- [1. Giới thiệu](#-gioi-thieu)
    - [1.1. Bài toán nhà xuất bản](#-bai-toan-nha-xuat-ban)
        - [Bài toán](#bai-toan)
        - [Phân tích](#phan-tich)
    - [1.2. Bài toán canh tác](#-bai-toan-canh-tac)
        - [Bài toán](#bai-toan-1)
        - [Phân tích](#phan-tich-1)
    - [1.3. Bài toán đóng thùng](#-bai-toan-dong-thung)
        - [Bài toán](#bai-toan-2)
        - [Phân tích](#phan-tich-2)
- [2. Nhắc lại bài toán tối ưu](#-nhac-lai-bai-toan-toi-uu)
    - [2.1. Các khái niệm cơ bản](#-cac-khai-niem-co-ban)
    - [2.2. Optimal and locally optimal points](#-optimal-and-locally-optimal-points)
    - [2.3. Một vài lưu ý](#-mot-vai-luu-y)
- [3. Bài toán tối ưu lồi](#-bai-toan-toi-uu-loi)
    - [3.1. Định nghĩa](#-dinh-nghia)
    - [3.2. Cực tiểu của bài toán tối ưu lồi chính là điểm tối ưu.](#-cuc-tieu-cua-bai-toan-toi-uu-loi-chinh-la-diem-toi-uu)
    - [3.3. Điều kiện tối ưu cho hàm mục tiêu khả vi](#-dieu-kien-toi-uu-cho-ham-muc-tieu-kha-vi)
    - [3.4. Giới thiệu thư viện CVXOPT](#-gioi-thieu-thu-vien-cvxopt)
- [4. Linear Programming](#-linear-programming)
    - [4.1. Dạng tổng quát của LP](#-dang-tong-quat-cua-lp)
    - [4.2. Dạng tiêu chuẩn của LP](#-dang-tieu-chuan-cua-lp)
    - [4.3. Minh hoạ bằng hình học của bài toán LP](#-minh-hoa-bang-hinh-hoc-cua-bai-toan-lp)
    - [Giải LP bằng CVXOPT](#giai-lp-bang-cvxopt)
- [5. Quadratic Programming](#-quadratic-programming)
    - [5.1. Định nghĩa bài toán Quadratic Programming](#-dinh-nghia-bai-toan-quadratic-programming)
    - [5.2. Ví dụ về QP](#-vi-du-ve-qp)
    - [5.3. Ví dụ về giải QP bằng CVXOPT](#-vi-du-ve-giai-qp-bang-cvxopt)
- [6. Geometric Programming](#-geometric-programming)
    - [6.1. Monomials và posynomials](#-monomials-va-posynomials)
    - [6.2. Geometric Programming](#-geometric-programming-1)
    - [6.3. Biến đổi GP về dạng convex](#-bien-doi-gp-ve-dang-convex)
    - [6.4. Giải GP bằng CVXOPT](#-giai-gp-bang-cvxopt)
- [7. Tóm tắt](#-tom-tat)
- [8. Tài liệu tham khảo](#-tai-lieu-tham-khao)

<!-- /MarkdownTOC -->

**Bạn được khuyến khích đọc [Bài 16](/2017/03/12/convexity/) trước khi đọc bài này. Nội dung trong bài viết này chủ yếu được dịch từ Chương 4 của cuốn _Convex Optimization_ trong phần Tài liệu tham khảo.**.

Bài này cũng có rất nhiều khái niệm mới và nhiều lý thuyết nên có thể không hấp dẫn như các bài khác. Tuy nhiên, tôi không thể bỏ qua vì không muốn các bạn hoàn toàn mất phương hướng khi đọc các bài sau.

Bạn đọc có thể xem bản pdf [tại đây](https://github.com/tiepvupsu/tiepvupsu.github.io/blob/master/assets/latex/book_CVX.pdf).

<a name="-gioi-thieu"></a>

## 1. Giới thiệu
Tôi xin bắt đầu bài viết này bằng ba bài toán khá gần với thực tế:
<a name="-bai-toan-nha-xuat-ban"></a>

### 1.1. Bài toán nhà xuất bản
<a name="bai-toan"></a>

#### Bài toán
Một nhà xuấn bản (NXB) nhận được đơn hàng 600 bản của cuốn "Machine Learning cơ bản" tới Thái Bình và 400 bản tới Hải Phòng. NXB đó có 800 cuốn ở kho Nam Định và 700 cuốn ở kho Hải Dương. Giá chuyển phát một cuốn sách từ Nam Định tới Thái Bình là 50,000 VND (50k), tới Hải Phòng là 100k. Giá chuyển phát một cuốn từ Hải Dương  tới Thái Bình là 150k, trong khi tới Hải Phòng chỉ là 40k. Hỏi để tốn ít chi phí chuyển phát nhất, công ty đó nên phân phối mỗi kho chuyển bao nhiêu cuốn tới mỗi địa điểm?

<a name="phan-tich"></a>

#### Phân tích
Để cho đơn giản, ta xây dựng bảng số lượng chuyển sách từ nguồn tới đích như sau:

|   Nguồn   |   Đích    | Đơn giá (\\(\times\\)10k) | Số lượng |
| :-------: | :-------: | :-----------------------: | :------: |
| Nam Định  | Thái Bình |             5             | \\(x\\)  |
| Nam Định  | Hải Phỏng |            10             | \\(y\\)  |
| Hải Dương | Thái Bình |            15             | \\(z\\)  |
| Hải Dương | Hải Phòng |             4             | \\(t\\)  |

Tổng chi phí (objective function) sẽ là \\(f(x, y, z, t) = 5x + 10y + 15z + 4t\\). Các điều kiện ràng buộc (constraints) viết dưới dạng biểu thức toán học là:

* Chuyển 600 cuốn tới Thái Bình: \\(x + z = 600\\).

* Chuyển 400 cuốn tới Hải Phòng: \\(y + t = 400\\).

* Lấy từ kho Nam Định không quá 800: \\(x + y \leq 800\\).

* Lấy từ kho Hải Dương không quá 700: \\(z + t \leq 700\\).

* \\(x, y, z, t\\) là các số tự nhiên. Ràng buộc là số tự nhiên sẽ khiến cho bài toán rất khó giải nếu số lượng biến là rất lớn. Với bài toán này, ta giả sử rằng \\(x, y, z, t\\) là các số thực dương. Khi tìm được nghiệm, nếu chúng không phải là số tự nhiên, ta sẽ lấy các giá trị tự nhiên gần nhất.

Vậy ta cần giải bài toán tối ưu sau đây:
<hr>
**Bài toán NXB:**
\\[
\begin{eqnarray}
    (x, y, z, t) =& \arg\min_{x, y, z, t} 5x + 10y + 15z + 4t ~~~~ (1)\\\
    \text{subject to:}~ & x + z = 600 ~~~~ (2)\\\
                        & y + t = 400 ~~~~ (3) \\\
                        & x + y \leq 800 ~~~(4) \\\
                        & z + t \leq 700 ~~~ (5)\\\
                        & x, y, z, t \geq 0 ~~~ (6)
\end{eqnarray}
\\]
<hr>
Nhận thấy rằng hàm mục tiêu (objective function) là một hàm tuyến tính của các biến \\(x, y, z, t\\). Các điều kiện ràng buộc đều có dạng _hyperplanes_ hoặc _haflspaces_, đều là các ràng buộc tuyến tính (linear constraints). Bài toán tối ưu với cả _objective function_ và _constraints_ đều là _linear_ được gọi là **Linear Programming (LP)**. Dạng tổng quát và cách thức lập trình để giải một bài toán thuộc loại này sẽ được cho trong phần sau của bài viết này.

_Nghiệm cho bài toán này có thể nhận thấy ngay là \\(x = 600, y = 0, z = 0, t = 400\\). Nếu ràng buộc nhiều hơn và số biến nhiều hơn, chúng ta cần một lời giải có thể tính được bằng cách lập trình._

<a name="-bai-toan-canh-tac"></a>

### 1.2. Bài toán canh tác
<a name="bai-toan-1"></a>

#### Bài toán
Một anh nông dân có tổng cộng 10ha (10 hecta) đất canh tác. Anh dự tính trồng cà phê và hồ tiêu trên số đất này với tổng chi phí cho việc trồng này là không quá 16T (triệu đồng). Chi phí để trồng cà phê là 2T cho 1ha, để trồng hồ tiêu là 1T/ha/. Thời gian trồng cà phê là 1 ngày/ha và hồ tiêu là 4 ngày/ha; trong khi anh chỉ có thời gian tổng cộng là 32 ngày. Sau khi trừ tất cả các chi phí (bao gồm chi phí trồng cây), mỗi ha cà phê mang lại lợi nhuận 5T, mỗi ha hồ tiêu mang lại lợi nhuận 3T. Hỏi anh phải trồng như thế nào để tối đa lợi nhuận? (_Các số liệu có thể vô lý vì chúng đã được chọn để bài toán ra nghiệm đẹp_)

<a name="phan-tich-1"></a>

#### Phân tích  
Gọi \\(x\\) và \\(y\\) lần lượt là số ha cà phê và hồ tiêu mà anh nông dân nên trồng. Lợi nhuận anh ấy thu được là \\(f(x, y) = 5x + 3y\\) (triệu đồng).

Các ràng buộc trong bài toán này là:

* Tổng diện tích trồng không vượt quá 10: \\(x + y \leq 10\\).

* Tổng chi phí trồng không vượt quá 16T: \\(2x + y \leq 16\\).

* Tổng thời gian trồng không vượt quá 32 ngày: \\(x + 4y \leq 32\\).

* Diện tích cà phê và hồ tiêu là các số không âm: \\(x, y \geq 0\\).

Vậy ta có bài toán tối ưu sau đây:
<hr>
**Bài toán canh tác:**
\\[
\begin{eqnarray}
    (x, y) =& \arg\max_{x, y} 5x + 3y ~~~~ (7)\\\
    \text{subject to:}~ & x + y \leq 10 ~~~~ (8)\\\
                        & 2x + y \leq 16 ~~~(9) \\\
                        & x + 4y \leq 32 ~~~ (10)\\\
                        & x, y \geq 0 ~~~ (11)
\end{eqnarray}
\\]
<hr>
Bài toán này hơi khác một chút là ta cần _tối đa hàm mục tiêu_ thay vì tối thiểu nó. Việc chuyển bài toán này về bài toán _tối thiểu_ có thể được thực hiện đơn giản bằng cách đổi dấu hàm mục tiêu. Khi đó hàm mục tiêu vẫn là _linear_, các ràng buộc vẫn là các _linear constraints_, ta lại có một bài toán **Linear Programming (LP)** nữa.

Bạn cũng có thể dựa vào hình minh hoạ dưới đây để suy ra nghiệm của bài toán:
<hr>
<div class="imgcap">
 <img src ="/assets/17_convexopt/planting.png" align = "center" width = "400">
 <div class = "thecap">Hình 1. Minh hoạ nghiệm cho bài toán canh tác.</div>
</div>
<hr>
Vùng màu xám có dạng _polyhedron_ (trong trường hợp này là đa giác) chính là tập hợp các điểm thoả mãn các ràng buộc từ \\(8)\\) đến \\((11)\\). Các đường nét đứt có màu chính là các đường đồng mức của hàm mục tiêu \\(5x + 3y\\), mỗi đường ứng với một giá trị khác nhau với đường càng đỏ ứng với giá trị càng cao. Một cách trực quan, nghiệm của bài toán có thể tìm được bằng cách di chuyển đường nét đứt màu xanh về phía bên phải (phía làm cho giá trị của hàm mục tiêu lớn hơn) đến khi nó không còn điểm chung với phần đa giác màu xám nữa.

Có thể nhận thấy nghiệm của bài toán chính là điểm màu xanh là giao điểm của hai đường thẳng \\(x + y = 10\\) và \\(2x + y = 16\\). Giải hệ phương trình này ta có \\(x^\* = 6\\) và \\(y^\* = 4\\). Tức anh nông dân nên trồng 6ha cà phê và 4ha hồ tiêu. Lúc đó lợi nhuận thu được là \\(5x^\* + 3y^\* = 42 \\) triệu đồng, trong khi anh chỉ mất thời gian là 22 ngày. (_Chịu tính toán cái là khác ngay, làm ít, hưởng nhiều_).

Đây chính là cách giải trong sách toán lớp 10 (ngày tôi học lớp 10).

Với nhiều biến hơn và nhiều ràng buộc hơn, chúng ta liệu có thể vẽ được hình như thế này để nhìn ra nghiệm hay không? Câu trả lời của tôi là nên tìm một công cụ để với nhiều biến hơn và với các ràng buộc khác nhau, chúng ta có thể tìm ra nghiệm gần như ngay lập tức.
<a name="-bai-toan-dong-thung"></a>

### 1.3. Bài toán đóng thùng
<a name="bai-toan-2"></a>

#### Bài toán
Một công ty phải chuyển 400 \\(m^3\\) cát tới địa điểm xây dựng ở bên kia sông bằng cách thuê một chiếc xà lan. Ngoài chi phí vận chuyển một lượt đi về là 100k của chiếc xà lan, công ty đó phải thiết kế một thùng hình hộp chữ nhật đặt trên xà lan để đựng cát. Chiếc thùng này không cần nắp, chi phí cho các mặt xung quanh là 1T/\\(m^2\\), cho mặt đáy là 2T/\\(m^2\\). Hỏi kích thước của chiếc thùng đó như thế nào để tổng chi phí vận chuyển là nhỏ nhất. Để cho đơn giản, giả sử cát chỉ được đổ ngang hoặc thấp hơn với phần trên của thành thùng, không có ngọn. Giả sử thêm rằng xà lan _rộng vô hạn_ và chứa được sức nặng vô hạn, giả sử này khiến bài toán dễ giải hơn.

<a name="phan-tich-2"></a>

#### Phân tích
Giả sử chiếc thùng cần làm có chiều dài là \\(x\\) (\\(m\\)), chiều rộng là \\(y\\) và chiều cao là \\(z\\). Thể tích của thùng là \\(xyz\\) (đơn vị là \\(m^3\\)). Có hai loại chi phí là:

* _Chi phí thuê xà lan:_ số chuyến xà lan phải thuê là \\(\frac{400}{xyz}\\) (ta hãy tạm giả sử rằng đây là một số tự nhiên, việc làm tròn này sẽ không thay đổi kết quả đáng kể vì chi phí vận chuyển một chuyến là nhỏ so với chi phí làm thùng). Số tiền phải trả cho xà lan sẽ là \\(0.1\frac{400}{xyz} = \frac{40}{xyz}\\).

* _Chi phí làm thùng:_ Diện tích xung quanh của thùng là \\(2 (x + y)z \\). Diện tích đáy là \\(xy\\). Vậy tổng chi phí làm thùng là \\(2(x +y)z + 2xy = 2(xy + yz + zx)\\).

Tổng toàn bộ chi phí là \\(f(x, y, z) = 40x^{-1}y^{-1}z^{-1} + 2(xy + yz + zx)\\). Điều kiện ràng buộc duy nhất là kích thước thùng phải là các số dương. Vậy ta có bài toán tối ưu sau:
<hr>
**Bài toán vận chuyển:**
\\[
\begin{eqnarray}
    (x, y) =& \arg\min_{x, y, z} 40x^{-1}y^{-1}z^{-1} + 2(xy + yz + zx) ~~~~ (13)\\\
    \text{subject to:}~ & x, y, z > 0 ~~~~ (14)\\\
\end{eqnarray}
\\]
<hr>
Bài toán này thuộc loại **Geometric Programming (GP)**. Định nghĩa của GP và cách dùng công cụ tối ưu sẽ được trình bày trong phần sau của bài viết.

_Nhận thấy rằng bài này hoàn toàn có thể dùng bất đẳng thức Cauchy để giải được, nhưng tôi vẫn muốn một lời giải cho bài toán tổng quát sao cho có thể lập trình được._

(Lời giải:
\\[
\begin{eqnarray}
    f(x, y, z) &=& \frac{20}{xyz} + \frac{20}{xyz} + 2xy + 2yz + 2zx \\\
               &\geq & 5\sqrt[5]{3200}
\end{eqnarray}
\\]
dấu bằng xảy ra khi và chỉ khi \\(x = y = z = \sqrt[5]{10}\\). Bài này có lẽ hợp với các kỳ thi vì dữ kiện quá đẹp. Cá nhân tôi thích các đề bài ra kiểu này hơn là yêu cầu đi tìm giá trị nhỏ nhất của một biểu thức nhàm chán, nhiều học sinh cho rằng không biết học bất đẳng thức để làm gì!)
<!-- http://www.pitt.edu/~jrclass/opt/notes6.pdf -->

Nếu có các ràng buộc về kích thước của thùng và trọng lượng mà xà lan tải được thì có thể tìm được lời giải đơn giản như thế này không?

Những bài toán trên đây đều là các bài toán tối ưu. Chính xác hơn nữa, chúng đều là các bài toán tối ưu lồi (_convex optimization problems_) như các bạn sẽ thấy ở phần sau. Và việc tìm lời giải có thể không mấy khó khăn, thậm chí giải bằng tay cũng có thể ra kết quả. Tuy nhiên, mục đích của bài viết này không phải là hướng dẫn các bạn giải các bài toán trên _bằng tay_, mà là cách nhận diện các bài toán và đưa chúng về các dạng mà các toolboxes sẵn có có thể giúp chúng ta. Trên thực tế, lượng dữ kiện và số biến cần tối ưu lớn hơn nhiều, chúng ta không thể giải các bài toán trên _bằng tay_ được.

Trước hết, chúng ta cần hiểu các khái niệm về _convex optimization problems_ và tại sao _convex_ lại quan trọng. (Bạn đọc có thể đọc tới [phần 4](/2017/03/19/convexopt/#-linear-programming) nếu không muốn biết các khái niệm và định lý toán  trong phần 2 và 3.)

<a name="-nhac-lai-bai-toan-toi-uu"></a>

## 2. Nhắc lại bài toán tối ưu
<a name="-cac-khai-niem-co-ban"></a>

### 2.1. Các khái niệm cơ bản
Tôi xin nhắc lại bài toán tối ưu ở dạng tổng quát:
\\[
\begin{eqnarray}
\mathbf{x}^* &=& \arg\min_{\mathbf{x}} f_0(\mathbf{x}) \\\
\text{subject to:}~ && f_i(\mathbf{x}) \leq 0, ~~ i = 1, 2, \dots, m ~~~(15)\\\
&& h_j(\mathbf{x}) = 0, ~~ j = 1, 2, \dots, p
\end{eqnarray}
\\]

Phát biểu bằng lời: Tìm giá trị của biến \\(\mathbf{x}\\) để tối thiểu hàm \\(f_0(\mathbf{x})\\) trong số các giá trị của \\(\mathbf{x}\\) thoả mãn các điệu hiện ràng buộc. Ta có bảng các tên gọi tiếng Anh và tiếng Việt như sau:

<hr>
| Ký hiệu                                  | Tiếng Anh                       | Tiếng Việt              |
| ---------------------------------------- | ------------------------------- | ----------------------- |
| \\(\mathbf{x} \in \mathbb{R}^n\\)        | optimization variable           | biến tối ưu             |
| \\(f_0: \mathbb{R}^n \rightarrow \mathbb{R}\\) | objective/los/cost function     | hàm mục tiêu            |
| \\(f_i(\mathbf{x}) \leq 0 \\)            | inequality constraints          | bất đẳng thức ràng buộc |
| \\(f_i: \mathbb{R}^n \rightarrow \mathbb{R}\\) | inequality constraint functions | -                       |
| \\(h_j(\mathbf{x}) = 0 \\)               | equality constraints            | đẳng thức ràng buộc     |
| \\(h_j: \mathbb{R}^n \rightarrow \mathbb{R}\\) | equality constraint functions   | -                       |
| \\(\mathcal{D} = \bigcap_{i=0}^m \text{dom}f_i \cap \bigcap_{pj=1}^p \text{dom}h_i \\) | domain                          | tập xác định            |

<hr>
Ngoài ra:

* Khi \\(m = p = 0\\), bài toán \\((15)\\) được gọi là _unconstrained optimization problem_ (bài toán tối ưu không ràng buộc).

* \\(\mathcal{D}\\) chỉ là tập xác định, tức giao của tất cả các tập xác định của mọi hàm số xuất hiện trong bài toán. Tập hợp các điểm thoả mãn mọi điều kiện ràng buộc, thông thường, là một tập con của \\(\mathcal{D}\\) được gọi là _feasible set_ hoặc _constraint set_. Khi _feasible set_ là một tập rỗng thì ta nói bài toán tối ưu \\((15)\\) là _infeasible_. Nếu một điểm nằm trong _feasible set_, ta gọi điểm đó là _feasible_.

* _Optimal value_ (_giá trị tối ưu_) của bài toán tối ưu \\((15)\\) được định nghĩa là:
  \\[
  p^* = \text{inf}\\{f\_0(\mathbf{x}) | f\_i(\mathbf{x}) \leq 0, i = 1, \dots, m; h_j(\mathbf{x}) = 0, j = 1, \dots, p\\}
  \\]
  trong đó \\(\text{inf}\\) là viết tắt của hàm [infimum](http://mathworld.wolfram.com/Infimum.html). \\(p^\*\\) có thể nhận các giá trị \\(\pm \infty\\). Nếu bài toán là _infeasible_, ta coi \\(p^* = + \infty\\), Nếu hàm mục tiêu không bị chặn dưới (_unbounded below_) trong tập xác định, ta coi \\(p^* = - \infty\\).


<a name="-optimal-and-locally-optimal-points"></a>

### 2.2. Optimal and locally optimal points
<!-- _Một vài khái niệm trong này các bạn có thể đã gặp trong chương trình toán cấp ba ở Việt Nam_. -->

Một điểm \\(\mathbf{x}^\*\\) được gọi là một điểm _optimal point_ (_điểm tối ưu_), hoặc là _nghiệm_ của bài toán \\((15)\\) nếu \\(\mathbf{x}^\*\\) là _feasible_ và \\(f_0(\mathbf{x}^*) = p^\*\\). Tất hợp tất cả các _optimal points_ được gọi là _optimal set_.

Nếu _optimal set_ là một tập _không_ rỗng, ta nói bài toán \\((15)\\) là _solvable_ (_giải được_). Ngược lại, nếu _optimal set_ là một tập rỗng, ta nói _optimal value_ là _không thể đạt được_ (_not attained/ not achieved_).

Ví dụ: xét hàm mục tiêu \\(f(x) = 1/x\\) với ràng buộc \\(x > 0\\). _Optimal value_ của bài toán này là \\(p^\* = 0\\) nhưng _optimal set_ là một tập rỗng vì không có giá trị nào của \\(x\\) để hàm mục tiêu đạt giá trị 0. Lúc này ta nói _giá trị tối ưu_ là _không đạt được_.

Với hàm một biến, một điểm là _cực tiểu_ của một hàm số nếu tại đó, hàm số đạt giá trị nhỏ nhất trong một lân cận (và lân cận này thuộc tập xác định của hàm số). Trong không gian 1 chiều, _lân cận_ được hiểu là trị tuyệt tối của hiệu 2 điểm nhỏ hơn một giá trị nào đó.

Trong toán tối ưu (thường là không gian nhiều chiều), ta gọi một điểm \\(\mathbf{x}\\) là __locally optimal__ (cực tiểu) nếu tồn tại một giá trị (thường được gọi là bán kinh) \\(R\\) sao cho:
\\[
\begin{eqnarray}
    f_0(\mathbf{x}) = &\text{inf}\\{f_0(\mathbf{z}) | f_i(\mathbf{z}) \leq 0, i = 1, \dots, m, \\\
                 & h_j(\mathbf{z}) = 0, j = 1, \dots, p, \|\|\mathbf{z} - \mathbf{x}\|\|_2 \leq R\\}
\end{eqnarray}
\\]

Nếu một điểm _feasible_ \\(\mathbf{x}\\) thoả mãn \\(f_i(\mathbf{x}) = 0\\), ta nói rằng bất đẳng thức ràng buộc thứ \\(i: f_i(\mathbf{x}) = 0\\) là _active_. Nếu \\(f_i(\mathbf{x}) < 0\\), ta nói rằng ràng buộc này là _inactive_ tại \\(\mathbf{x}\\).

<a name="-mot-vai-luu-y"></a>

### 2.3. Một vài lưu ý
Mặc dù trong định nghĩa bài toán tối ưu \\((15)\\) là cho bài toán _tối thiểu hàm mục tiêu_ với các ràng buộc thoả mãn các điều kiện nhỏ hơn hoặc bằng 0, các bài toán tối ưu với _tối đa hàm mục tiêu_ và điều kiện ràng buộc ở dạng khác đều có thể đưa về được dạng này:

* \\(\max f_0(\mathbf{x}) \Leftrightarrow\min -f_0(\mathbf{x}) \\).

* \\(f_i(\mathbf{x}) \leq g(\mathbf{x}) \Leftrightarrow\ f_i(\mathbf{x}) - g(\mathbf{x}) \leq 0\\).

* \\(f_i(\mathbf{x}) \geq 0 \Leftrightarrow\ -f_i(\mathbf{x}) \leq 0 \\).

* \\(a \leq f_i(\mathbf{x}) \leq b \Leftrightarrow\ f_i(\mathbf{x}) -b \leq 0\\) và \\(a - f_i(\mathbf{x}) \leq 0\\).

* \\(f_i(\mathbf{x}) \leq 0 \Leftrightarrow f_i(\mathbf{x}) + s_i = 0 \\) và \\(s_i \geq 0\\). \\(s_i\\) được gọi là _slack variable_. Phép biến đổi đơn giản này trong nhiều trường hợp lại tỏ ra hiệu quả vì bất đẳng thức \\(s_i \geq 0\\) thường dễ giải quyết hơn là \\(f_i(\mathbf{x}) \leq 0\\).

<a name="-bai-toan-toi-uu-loi"></a>

## 3. Bài toán tối ưu lồi

Trong toán tối ưu, chúng ta đặc biệt quan tâm tới những bài toán mà hàm mục tiêu là một hàm lồi, và _feasible set_ cũng là một tập lồi.
<a name="-dinh-nghia"></a>

### 3.1. Định nghĩa
Một _bài toán tối ưu lồi_ (_convex optimization problem_) là một bài toán tối ưu có dạng:
\\[
\begin{eqnarray}
\mathbf{x}^* &=& \arg\min_{\mathbf{x}} f_0(\mathbf{x}) \\\
\text{subject to:}~ && f_i(\mathbf{x}) \leq 0, ~~ i = 1, 2, \dots, m ~~~(16)\\\
&& \mathbf{a}_j^T\mathbf{x} - b_j = 0, j = 1, \dots,
\end{eqnarray}
\\]
trong đó \\(f_0, f_1, \dots, f_m\\) là các hàm lồi.

So với bài toán tối ưu \\((15)\\), bài toán tối ưu lồi \\((16)\\) có thêm ba điều kiện nữa:

* _Hàm mục tiêu_ là một _hàm lồi_.

* Các hàm bất đẳng thức ràng buộc \\(f_i\\) là các hàm lồi.

* Hàm đẳng thức ràng buộc \\(h_j\\) là _affine_ (hàm _linear_ cộng với một hẳng số nữa được gọi là _affine_).

Một vài nhận xét:

* Tập hợp các điểm thoả mãn \\(h_j(\mathbf{x}) = 0\\) là một tập lồi vì nó có dạng một _hyperplane_.

* Khi \\(f_i\\) là một _hàm lồi_ thì tập hợp các điểm thoả mãn \\(f_i(\mathbf{x}) \leq 0 \\) chính là [0-sublevel set của \\(f_i\\) và là một tập lồi](/2017/03/12/convexity/#-\\\alpha-\\-sublevel-sets).

* Như vậy tập hợp các điểm thoả mãn mọi điều kiện ràng buộc chính là [giao điểm của các _tập lồi_, vì vậy nó là một _tập lồi_](/2017/03/12/convexity/#-giao-cua-cac-tap-loi-la-mot-tap-loi).

**Vậy, trong một bài toán tối ưu lồi, ta _tối thiểu một hàm mục tiêu lồi_ trên một _tập lồi_**.

<a name="-cuc-tieu-cua-bai-toan-toi-uu-loi-chinh-la-diem-toi-uu"></a>

### 3.2. Cực tiểu của bài toán tối ưu lồi chính là điểm tối ưu.
TÍnh chất quan trọng nhất của bài toán tối ưu lồi chính là bất kỳ _locally optimal point_ chính là một điểm _(globally) optimal point_.

Tính chất quan trọng này có thể chứng minh bằng phản chứng như sau. Gọi  \\(\mathbf{x}\_0\\) là một điểm _locally optimal_, tức:

\\[
f\_0(\mathbf{x}\_0) = \text{inf} \\{f\_0(\mathbf{x}) | \mathbf{x} ~\text{is feasible}, \|\|\mathbf{x} - \mathbf{x}\_0\|\|_2 \leq R\\}
\\]

với \\(R > 0\\) nào đó. Giả sử \\(\mathbf{x}\_0\\) không phải là _globally optimal point_, tức tồn tại một _feasible point_ \\(\mathbf{y}\\) sao cho \\(f(\mathbf{y}) < f(\mathbf{x}\_0)\\) (hiển nhiên rằng \\(\mathbf{y}\\) không nằm trong lân cận đang xét). Ta có thể tìm được \\(\theta \in [0, 1]\\) đủ nhỏ sao cho \\(\mathbf{z} = (1 - \theta)\mathbf{x}\_0 + \theta\mathbf{y}\\) nằm trong lân cận của \\(\mathbf{x}\_0\\), tức \\(\|\|\mathbf{z} - \mathbf{x}\_0\|\|\_2 < R\\). Chú ý rằng \\(\mathbf{z}\\) cũng là một _feasible point_ vì _feasible set_ là một _tập lồi_. Hơn nữa, vì _hàm mục tiêu_ \\(f_0\\) là một hàm lồi, ta có:

\\[
\begin{eqnarray}
    f\_0(\mathbf{z}) &=& f\_0((1 - \theta)\mathbf{x}\_0 + \theta \mathbf{y})  \\\
                    &\leq& (1 - \theta)f\_0(\mathbf{x}\_0) + \theta f\_0(\mathbf{y})\\\
                    & < & (1 - \theta)f\_0(\mathbf{x}\_0) + \theta f\_0(\mathbf{x}\_0) \\\
                    &=& f\_0(\mathbf{x}\_0)
\end{eqnarray}
\\]

điều này mâu thuẫn với giả thiết \\(\mathbf{x}\_0\\) là một điểm cực tiểu. Vậy giả sử sai, tức \\(\mathbf{x}\_0\\) chính là _globally optimal point_ và ta có điều phải chứng minh.

Chứng minh bằng lời: giả sử một điểm cực tiểu không phải là điểm làm cho hàm số đạt giá trị nhỏ nhất. Với điều kiện _feasible set_ và _hàm mục tiêu_ là lồi, ta luôn tìm được một điểm khác trong lân cận của điểm cực tiểu đó sao cho giá trị của hàm mục tiêu tại điểm mới này nhỏ hơn giá trị của hàm mục tiêu tại điểm cực tiểu. Sự mâu thuẫn này chỉ ra rằng với một bài toán tối ưu lồi, điểm cực tiểu phải là điểm làm cho hàm số đạt giá trị nhỏ nhất.

<a name="-dieu-kien-toi-uu-cho-ham-muc-tieu-kha-vi"></a>

### 3.3. Điều kiện tối ưu cho hàm mục tiêu khả vi

Nếu hàm mục tiêu \\(f_0\\) là khả vi (differentiable), theo [first-order condition](/2017/03/12/convexity/#-first-order-condition), với mọi \\(\mathbf{x}, \mathbf{y} \in \text{dom}f_0\\), ta có:
\\[
f\_0(\mathbf{x}) \geq f\_0(\mathbf{x}\_0) + \nabla f_0(\mathbf{x}\_0)^T (\mathbf{x} - \mathbf{x}\_0)~~~(17)
\\]

Đặt \\(\mathcal{X}\\) là _feasible set_. **Điều kiện cần và đủ** để một điểm \\(\mathbf{x}\_0 \in \mathcal{X}\\) là _optimal point_ là:
\\[
\nabla f_0(\mathbf{x}_0)^T(\mathbf{x} - \mathbf{x}\_0) \geq 0, ~\forall \mathbf{x} \in \mathcal{X} ~~~(18)
\\]
Tôi xin được bỏ qua việc chứng minh điều kiện cần và đủ này, bạn đọc có thể tìm trong trang 139-140 của cuốn Convex Optimization trong Tài liệu tham khảo.

Xem hình vẽ dưới đây:
<hr>
<div class="imgcap">
 <img src ="/assets/17_convexopt/optimalitycondition.png" align = "center" width = "600">
 <div class = "thecap">Hình 2. Biểu diễn hình học của điều kiện tối ưu cho hàm mục tiêu khả vi. Các đường nét đứt có màu tương ứng với các level sets (đường đồng mức).</div>
</div>
<hr>
Một cách hình học, điều kiện này nói rằng: Nếu \\(\mathbf{x}\_0\\) là điểm _optimal_ thì với mọi \\(\mathbf{x} \in \mathcal{X}\\), vector đi từ \\(\mathbf{x}\_0\\) tới \\(\mathbf{x}\\) hợp với vector \\(-\nabla f_0 (\mathbf{x}\_0)\\) một góc tù. Nói cách khác, nếu ta vẽ _mặt tiếp tuyến_ của hàm mục tiêu tại \\(\mathbf{x}\_0\\) thì mọi điểm _feasible_ nằm về một phía so với _mặt tiếp tuyến này_. Hơn nữa, _feasible set_ nằm về phía làm cho hàm mục tiêu đạt giá trị cao hơn \\(f_0(\mathbf{x}\_0)\\). Mặt tiếp tuyến này chính là _supporting hyperplane_ của _feasible set_ tại điểm \\(\mathbf{x}\_0\\). Nhắc lại rằng khi vẽ các _level set_, tôi thường dùng màu lam để chỉ giá trị nhỏ, màu đỏ để chỉ giá trị lớn của hàm số.

(Một mặt phẳng đi qua một điểm trên biên của một tập hợp sao cho mọi điểm trong tập hợp đó nằm về một phía (hoặc nằm trên) so với mặt phẳng đó được gọi là _supporting hyperplane_ (_siêu phẳng hỗ trợ_). Nếu một tập hợp là _lồi_, tồn tại _supporting hyperplane_ tại mọi điểm trên biên của nó.)

Nếu tồn tại một điểm \\(\mathbf{x}\_0\\) trong _feasible set_ sao cho \\(\nabla f_0(\mathbf{x}\_0) = 0\\), đây chính là _optimal point_. Điều này dễ hiểu vì đó chính là điểm làm cho gradient bằng 0, tức điểm cực tiểu của hàm mục tiêu. Nếu \\(\nabla f_0(\mathbf{x}\_0) \neq 0\\), vector \\(-\nabla f_0 (\mathbf{x}\_0)\\) chính là _vector pháp tuyến_ của _supporting hyperplane_ tại \\(\mathbf{x}\_0\\).

<a name="-gioi-thieu-thu-vien-cvxopt"></a>

### 3.4. Giới thiệu thư viện CVXOPT
[CVXOPT](http://cvxopt.org/) là một thư viện miễn phí trên Python giúp giải rất nhiều các bài toán trong cuốn sách Convex Optimization ở phần Tài liệu tham khảo. Tác giả thứ hai của cuốn sách này, Lieven Vandenberghe, chính là đồng tác giả của thư viện này. Hướng dẫn cài đặt, tài liệu hướng dẫn, và các ví dụ mẫu của thư viện này cũng có đầy đủ trên trang web [CVXOPT](http://cvxopt.org/).

Trong phần còn lại của bài viết, tôi sẽ giới thiệu 3 bài toán rất cơ bản trong Convex Optimization: Linear Programming, Quadratic Programming, và Geometric Programming. Tôi cũng sẽ cùng các bạn lập trình để giải các ví dụ đã nêu ở phần đầu bài viết dựa trên thư viện CVXOPT này.


<a name="-linear-programming"></a>

## 4. Linear Programming
Chúng ta cùng bắt đầu với lớp các bài toán đơn giản nhất trong Convex Optimization - Linear Programming (LP, một số tài liệu cũng gọi là _Linear Program_), trong đó hàm mục tiêu \\(f_0\\) và hàm bất đẳng thức ràng buộc \\(f_i, i = 1, \dots, m\\) đều là các hàm tuyến tính cộng với một hằng số (tức [hàm _affine_](/2017/03/19/convexopt/#-linear-programming)).

<a name="-dang-tong-quat-cua-lp"></a>

### 4.1. Dạng tổng quát của LP
<hr>
**A general LP:**
\\[
\begin{eqnarray}
\mathbf{x} &=& \arg\min_{\mathbf{x}} \mathbf{c}^T\mathbf{x} + d \\\
\text{subject to:}~ && \mathbf{Gx} \preceq \mathbf{h} ~~~~~~~~~~~~~~~~~~~~(19)\\\
&& \mathbf{Ax} = \mathbf{b}
\end{eqnarray}
\\]
<hr>
Trong đó \\(\mathbf{G} \in \mathbb{R}^{m\times n}, \mathbf{h} \in \mathbb{R}^m\\) và, \\(\mathbf{A}\in \mathbb{R}^{p\times n}, \mathbf{b} \in\mathbb{R}^p\\). \\(\mathbf{c}, \mathbf{x} \in\mathbb{R}^n\\) và \\(d\\) là một số vô hướng (số vô hướng này có thể bỏ qua vì nó không ảnh hưởng tới nghiệm của bài toán tối ưu, nó chỉ làm thay đổi giá trị của hàm mục tiêu). Nhắc lại rằng ký hiệu \\(\preceq\\) nghĩa là mỗi phần tử trong vector (ma trận) ở vế trái nhỏ hơn hoặc bằng phần tử tương ứng trong vector (ma trân) ở về phải.

Chú ý rằng nhiều bất đẳng thức dạng \\(\mathbf{g}\_i\mathbf{x} \leq h\_i\\), với \\(\mathbf{g}\_i\\) là các vector hàng, có thể viết gộp dưới dạng \\(\mathbf{Gx} \preceq \mathbf{h}\\) trong đó mỗi hàng của \\(\mathbf{G}\\) ứng với một \\(\mathbf{g}\_i\\), mỗi phần tử của \\(\mathbf{h}\\) tương ứng với một \\(h_i\\).

<a name="-dang-tieu-chuan-cua-lp"></a>

### 4.2. Dạng tiêu chuẩn của LP
Trong dạng tiêu chuẩn (_standard form_) LP, các bất đẳng thức ràng buộc chỉ là điều kiện các nghiệm có thành phần không âm:
<hr>
**A standard form LP:**
\\[
\begin{eqnarray}
\mathbf{x} &=& \arg\min_{\mathbf{x}} \mathbf{c}^T\mathbf{x} \\\
\text{subject to:}~ && \mathbf{Ax} = \mathbf{b} ~~~~~~~~~~~~~~~~~~~~(20)\\\
&& \mathbf{x} \succeq \mathbf{0}
\end{eqnarray}
\\]
<hr>

Bài toán \\((19)\\) có thể đưa về bài toán \\((20)\\) bằng cách đặt thêm biến _slack_ \\(\mathbf{s}\\)
<hr>
\\[
\begin{eqnarray}
\mathbf{x} &=& \arg\min_{\mathbf{x}, \mathbf{s}} \mathbf{c}^T\mathbf{x} \\\
\text{subject to:}~ && \mathbf{Ax} = \mathbf{b} ~~~~~~~~~~~~~~~~~~~~(21)\\\
&& \mathbf{Gx} + \mathbf{s} = \mathbf{h} \\\
&& \mathbf{s} \succeq \mathbf{0}
\end{eqnarray}
\\]
<hr>
Tiếp theo, nếu ta biểu diễn \\(\mathbf{x}\\) dưới dạng hiệu của hai vector mà thành phần của nó đều không âm, tức: \\(\mathbf{x} = \mathbf{x}^+ - \mathbf{x}^-\\), với \\(\mathbf{x}^+, \mathbf{x}^- \succeq 0\\). Ta có thể tiếp tục viết lại \\((21)\\) dưới dạng:
<hr>
\\[
\begin{eqnarray}
\mathbf{x} &=& \arg\min_{\mathbf{x}^+,\mathbf{x}^-, \mathbf{s}} \mathbf{c}^T\mathbf{x}^+ - \mathbf{c}^T\mathbf{x}^- \\\
\text{subject to:}~ && \mathbf{Ax}^+ - \mathbf{Ax}^- = \mathbf{b} ~~~~~~~~~~~~~~~~~~~~(22)\\\
&& \mathbf{Gx}^+ - \mathbf{Gx}^- + \mathbf{s} = \mathbf{h} \\\
&& \mathbf{x}^+ \succeq 0, \mathbf{x}^- \succeq 0, \mathbf{s} \succeq \mathbf{0}
\end{eqnarray}
\\]
<hr>
Tới đây, bạn đọc có thể thấy rằng \\((22)\\) có thể viết gọn lại như \\((20)\\).



Bài toán nhà xuất bản và Bài toán canh tác trong phần đầu của bài viết này chính là các LP.

<a name="-minh-hoa-bang-hinh-hoc-cua-bai-toan-lp"></a>

### 4.3. Minh hoạ bằng hình học của bài toán LP
 Các bài toán LP có thể được minh hoạ như hình dưới đây:

<hr>
<div class="imgcap">
 <img src ="/assets/17_convexopt/lp.png" align = "center" width = "600">
 <div class = "thecap">Hình 3. Biểu diễn hình học của Linear Programming.</div>
</div>
<hr>

Điểm \\(\mathbf{x}\_0\\) chính là điểm làm cho hàm mục tiêu đạt giá trị nhỏ nhất, điểm \\(\mathbf{x}\_1\\) chính là điểm làm cho hàm mục tiêu đạt giá trị lớn nhất. Với các bài toán LP, nghiệm, nếu có, thường là một điểm ở _đỉnh_ của polyheron hoặc là một _mặt_ của polyhedron đó (trong trường hợp các đường level sets của hàm mục tiêu song song với mặt đó, và trên mặt đó, hàm mục tiêu đạt giá trị tối ưu).

Về LP, các bạn có thể tìm thấy rất nhiều tài liệu cả tiếng Việt (Quy hoạch tuyến tính) và tiếng Anh. Có rất nhiều các bài toán trong thực tế có thể đưa về dạng LP. Phương pháp thường được dùng để giải bài toán này có tên là _simplex_ (_đơn hình_). Tôi sẽ không đề cập đến các phương pháp này, thay vào đó, tôi sẽ hướng dẫn các bạn dùng thư viện CVXOPT để giải quyết các bài toán thuộc dạng này.

<a name="giai-lp-bang-cvxopt"></a>

### Giải LP bằng CVXOPT

Tôi sẽ dùng thư viện CVPOPT để giải Bài toán canh tác ở phía trên. Nhắc lại bài toán canh tác:
<hr>
**Bài toán canh tác:**
\\[
\begin{eqnarray}
(x, y) =& \arg\max_{x, y} 5x + 3y \\\
\text{subject to:}~ & x + y \leq 10 \\\
                    & 2x + y \leq 16  \\\
                    & x + 4y \leq 32 \\\
                    & x, y \geq 0
\end{eqnarray}
\\]
<hr>

Các điều kiện ràng buộc có thể viết lại dưới dạng \\( \mathbf{Gx} \preceq \mathbf{h}\\), trong đó:

\\[
\mathbf{G} = \left\[\begin{matrix}
1 & 1 \\\
2 & 1 \\\
1 & 4 \\\
-1 & 0 \\\
0 & -1
\end{matrix}\right\], ~~~~
\mathbf{h} = \left\[\begin{matrix}
10\\\
16 \\\
32 \\\
0 \\\
0
\end{matrix}\right\]
\\]

Lời giải cho bài toán này khi dùng CVXOPT là:
```python
from cvxopt import matrix, solvers
c = matrix([-5., -3.])
G = matrix([[1., 2., 1., -1., 0.], [1., 1., 4., 0., -1.]])
h = matrix([10., 16., 32., 0., 0.])

solvers.options['show_progress'] = False
sol = solvers.lp(c, G, h)

print('Solution"')
print(sol['x'])
```

```
Solution:
[ 6.00e+00]
[ 4.00e+00]
```
Nghiệm này chính là nghiệm mà tôi đã tìm được trong phần đầu của bài viết.

Một vài lưu ý:

* Hàm `solvers.lp` của `cvxopt` giải bài toán \\((21)\\).

* Trong bài toán của chúng ta, vì ta cần tìm giá trị lớn nhất nên ta phải đổi hàm mục tiêu về dạng \\(-5x - 3y\\). Chính vì vậy mà `c = matrix([-5., -3.])`.

* Hàm `matrix` nhận đầu vào là một `list` (trong Python), `list` này thể hiện một vector cột. Nếu muốn biểu diễn một ma trận, đầu vào của `matrix` là một `list` của `list`, trong đó mỗi `list` bên trong thể hiện một vector cột của ma trận đó.

* Các hằng số trong bài toán cần ở dạng số thực. Nếu chúng là các số nguyên, ta cần thêm dấu `.` vào sau các số đó thể thể hiện đó là số thực. (Tôi thấy điểm này hơi thừa, nhưng nếu không có dấu `.` thì chương trình sẽ báo lỗi.)

* Với đẳng thức ràng buộc \\(\mathbf{Ax} = \mathbf{b}\\), `solvers.lp` lấy giá trị mặc định của `A` và `b` là `None`, tức nếu không khái báo thì nghĩa là không có đẳng thức ràng buộc nào.

* Với các tuỳ chọn khác, bạn đọc có thể tìm trong Tài liệu của CVXOPT.

Việc giải Bài toán nhà xuất bản bằng CVXOPT xin nhường lại cho bạn đọc như một bài tập đơn giản.

<a name="-quadratic-programming"></a>

## 5. Quadratic Programming
<a name="-dinh-nghia-bai-toan-quadratic-programming"></a>

### 5.1. Định nghĩa bài toán Quadratic Programming
Một dạng Convex Optimization mà các bạn sẽ gặp rất nhiều trong các bài sau của blog là _Quadratic Programming_ (QP, hoặc _Quadratic Program_). Khác biệt duy nhất của QP so với LP là hàm mục tiêu có dạng _Quadratic form_:

<hr>
\\[
\begin{eqnarray}
\mathbf{x} &=& \arg\min_{\mathbf{x}} \frac{1}{2} \mathbf{x}^T\mathbf{P}\mathbf{x} + \mathbf{q}^T\mathbf{x} + \mathbf{r} \\\
\text{subject to:} &&\mathbf{Gx} \preceq \mathbf{h} ~~~~~~~~~~~~~(23)\\\
&& \mathbf{Ax} = \mathbf{b}
\end{eqnarray}
\\]
<hr>
Trong đó \\(\mathbf{P} \in \mathbb{S}\_+^n\\) (tập các ma trận vuông nửa xác định dương có số cột là \\(n\\)), \\(\mathbf{G}\in \mathbb{R}^{m\times n}, \mathbf{A}\in\mathbb{R}^{p \times n}\\). [Điều kiện \\(\mathbf{P}\\) là _nửa xác định dương_ để đảm bảo hàm mục tiêu là _convex_](http://machinelearningcoban.com/2017/03/12/convexity/#-quadratic-forms).

Chúng ta có thể thấy rằng LP chính là một trường hợp đặc biệt của QP với \\(\mathbf{P} = \mathbf{0}\\).

Diễn đạt bằng lời: trong QP, chúng ta tối thiểu một hàm quadratic lồi trên một _polyhedron_. Xem hình dưới đây:
<hr>
<div class="imgcap">
 <img src ="/assets/17_convexopt/qp.png" align = "center" width = "600">
 <div class = "thecap">Hình 4. Biểu diễn hình học của Quadratic Programming.</div>
</div>
<hr>

<a name="-vi-du-ve-qp"></a>

### 5.2. Ví dụ về QP
Bài toán vui: Có một hòn đảo mà hình dạng của nó có dạng một đa giác lồi. Một con thuyền ở ngoài biển thì cần đi theo hướng nào để tới đảo nhanh nhất, giả sử rằng tốc độ của sóng và gió bằng 0.

<!-- (_polyhedron_ - đa giác trong không gian nhiều chiều, _polyhedra_ - số nhiều của _polyhedron_.) -->

Bài toán khoảng cách từ một điểm tới một polyhedron được phát biểu như sau:

Cho một polyhedron được biểu diễn bởi \\(\mathbf{Ax} \preceq \mathbf{b}\\) và một điểm \\(\mathbf{u}\\), tìm điểm \\(\mathbf{x}\\) thuộc polyhedron đó sao cho khoảng cách Euclidean giữa \\(\mathbf{x}\\) và \\(\mathbf{u}\\) là nhỏ nhất.

Bài toán này có thể phát biểu như sau:
\\[
\begin{eqnarray}
\mathbf{x} &=& \arg\min_{\mathbf{x}} \frac{1}{2}\|\|\mathbf{x} - \mathbf{u}\|\|\_2^2 \\\
\text{subject to:} &&\mathbf{Gx} \preceq \mathbf{h}\\\
\end{eqnarray}
\\]
Hàm mục tiêu đạt giá trị nhỏ nhất bằng 0 nếu \\(\mathbf{u}\\) nằm trong polyheron đó và _optimal point_ chính là \\(\mathbf{x} = \mathbf{u}\\). Khi \\(\mathbf{u}\\) không nằm trong polyhedron, ta viết:
\\[
\frac{1}{2} \|\|\mathbf{x} - \mathbf{u}\|\|\_2^2 = \frac{1}{2} (\mathbf{x} - \mathbf{u})^T(\mathbf{x} - \mathbf{u}) = \frac{1}{2} \mathbf{x}^T\mathbf{x} - \mathbf{u}^T\mathbf{x} + \frac{1}{2} \mathbf{u}^T\mathbf{u}
\\]

Biểu thức này có dạng hàm mục tiêu như trong \\((23)\\) với \\(\mathbf{P = I}, \mathbf{q} = - \mathbf{u}, \mathbf{r} = \frac{1}{2} \mathbf{u}^T\mathbf{u}\\), trong đó \\(\mathbf{I}\\) là ma trận đơn vị.

<a name="-vi-du-ve-giai-qp-bang-cvxopt"></a>

### 5.3. Ví dụ về giải QP bằng CVXOPT

Xét bài toán sau đây:

\\[
\begin{eqnarray}
(x, y) &=& \arg\min_{x, y} (x - 10)^2 + (y - 10)^2 \\\
\text{subject to:}~&&
\left\[\begin{matrix}
1 & 1 \\\
2 & 1 \\\
1 & 4 \\\
-1 & 0 \\\
0 & -1
\end{matrix}\right\]
\left\[
\begin{matrix}
x \\\
y
\end{matrix}
\right\]
\preceq
\left\[\begin{matrix}
10\\\
16 \\\
32 \\\
0 \\\
0
\end{matrix}\right\]
\end{eqnarray}
\\]

<hr>
<div class="imgcap">
 <img src ="/assets/17_convexopt/qp_ex.png" align = "center" width = "400">
 <div class = "thecap">Hình 5. Ví dụ về khoảng cách giữa một điểm và một polyhedron.</div>
</div>
<hr>

_Feasible set_ trong bài toán này tôi lấy trực tiếp từ Bài toán canh tác và \\(\mathbf{u} = [10, 10]^T\\).
Bài toán này có thể được giải bằng CVXOPT như sau:

```python
from cvxopt import matrix, solvers
P = matrix([[1., 0.], [0., 1.]])
q = matrix([-10., -10.])
G = matrix([[1., 2., 1., -1., 0.], [1., 1., 4., 0., -1.]])
h = matrix([10., 16., 32., 0., 0])

solvers.options['show_progress'] = False
sol = solvers.qp(P, q, G, h)

print('Solution:')
print(sol['x'])
```
```
Solution:
[ 5.00e+00]
[ 5.00e+00]
```

Trong các thuật toán Machine Learning, các bạn sẽ gặp các bài toán về tìm _hình chiếu_ (projection) của một điểm lên một _tập lồi_ nói chung rất nhiều. Tới từng phần, tôi sẽ đề cập hướng giải quyết của các bài toán đó.


<a name="-geometric-programming"></a>

## 6. Geometric Programming
Trong mục này, chúng ta sẽ thấy một lớp các bài toán _không lồi_ khi nhìn vào hàm mục tiêu và các hàm ràng buộc, nhưng có thể được biến đổi về dạng _lồi_ bằng một vài kỹ thuật.

Trước hết, chúng ta cần có một vài định nghĩa:

<a name="-monomials-va-posynomials"></a>

### 6.1. Monomials và posynomials

Một hàm số \\(f: \mathbb{R}^n \rightarrow \mathbb{R}\\) với tập xác đinh \\(\text{dom}f = \mathbf{R}\_{++}^n\\) (tất cả các phần tử đều là số dương) có dạng:
\\[
f(\mathbf{x}) = c x_1^{a_1} x_2^{a_2} \dots x_n^{a_n}~~~~~~~~(24)
\\]
trong đó \\(c > 0\\) và \\(a\_i \in \mathbb{R}\\), được gọi là một _monomial function_ (khái niệm này khá giống với _đơn thức_ khi tôi học lớp 8, nhưng khi đó SGK định nghĩa với \\(c\\) bất kỳ và \\(a\_i\\) là các số tự nhiên).

Tổng của các monomials:
\\[
f(\mathbf{x}) = \sum_{k=1}^K c_k x_1^{a_{1k}}x_2^{a_{2k}}\dots x_n^{a_{nk}}~~~~~~~~~~(24)
\\]
trong đó các \\(c\_k > 0\\) được gọi là _posynomial function_ (_đa thức_), hoặc đơn giản là _posynomial_.

<a name="-geometric-programming-1"></a>

### 6.2. Geometric Programming

Một bài toán tối ưu có dạng:

<hr>
\\[
\begin{eqnarray}
\mathbf{x}   &=& \arg\min_{\mathbf{x}} f_0(\mathbf{x}) \\\
\text{subject to:}~ && f_i(x) \leq 1,  ~~ i = 1, 2, \dots, m ~~~~~~~~~~~(25)\\\
                    && h_j(x) = 1, ~~ j = 1, 2, \dots, p
\end{eqnarray}
\\]
<hr>

trong đó \\(f_0, f_1, \dots, f_m\\) là các _posynomials_ và \\(h_1, \dots, h_p\\) là các _monomials_, được gọi là _Geometric Programming_ (GP). Điều kiện \\(\mathbf{x} \succ 0\\) được ẩn đi.

Chú ý rằng nếu \\(f\\) là một _posynomial_, \\(h\\) là một _monomial_ thì \\(f/h\\) là một _posynomial_.

**Ví dụ:**
\\[
\begin{eqnarray}
    (x, y, z)    &=& \arg\min_{x, y, z} x/y                          \\\
\text{subject to:}~ && 1 \leq x \leq 2 \\\
 && x^3 + 2y/z \leq \sqrt{y} \\\
 && x/y = z
\end{eqnarray}
\\]
Có thể được viết lại dưới dạng GP:
\\[
\begin{eqnarray}
    (x, y, z)    &=& \arg\min_{x, y, z} xy ^{-1}                        \\\
\text{subject to:}~ && x^{-1} \leq 1 \\\
&& (1/2)x \leq 1 \\\
&& x^3y^{-1/2} + 2y^{1/2}z^{-1} \leq 1 \\\
&& xy^{-1}z^{-1} = 1
\end{eqnarray}
\\]
Bài toán này rõ ràng là _nonconvex_ vì cả hàm mục tiêu và điều kiển ràng buộc đều không lồi.

<a name="-bien-doi-gp-ve-dang-convex"></a>

### 6.3. Biến đổi GP về dạng convex
GP có thể được biến đổi về dạng lồi như sau:

Đặt \\(y_i = \log(x_i)\\), tức \\(x_i = \exp({y_i})\\). Nếu \\(f\\) là một _monomial function_ của \\(\mathbf{x}\\) thì:
\\[
f(\mathbf{x}) = c(\exp({y_1}))^{a_1} \dots (\exp({y_n}))^{a_n} = \exp({\mathbf{a}^T\mathbf{y} + b})
\\]
với \\(b = \log(c)\\). Lúc này, hàm số \\(g(y) = \exp({\mathbf{a}^T\mathbf{y} + b})\\) là một hàm lồi theo \\(\mathbf{y}\\). (Bạn đọc có thể chứng minh theo định nghĩa rằng hợp của hai hàm lồi là một hàm lồi. Trong trường hợp này, hàm \\(\exp\\) và hàm _affine_ trên đều là các hàm lồi.)

Tương tự như thế, _posynomial_ trong đẳng thức \\((24)\\) có thể viết dưới dạng:
\\[
f(\mathbf{x}) = \sum_{k = 1}^K \exp(\mathbf{a}\_k^T\mathbf{y} + b\_k)
\\]
trong đó \\(\mathbf{a}\_k = [a_{1k}, \dots, a_{nk}]^T\\) và \\(b_k = \log(c_k)\\). Lúc này, _posynomial_ đã được viết dưới dạng tổng của các hàm \\(\exp\\) của các hàm _affine_ (và vì vậy là một hàm lồi, nhớ lại rằng tổng của các hàm lồi là một hàm lồi).

Bài toán GP \\((25)\\) được viết lại dưới dạng:
\\[
\begin{eqnarray}
    \mathbf{y}    &=& \arg\min_{\mathbf{y}} \sum_{k=1}^{K_0} \exp(\mathbf{a}\_{0k}^T\mathbf{y} + b\_{0k})                      \\\
\text{subject to:}~ && \sum_{k=1}^{K_i} \exp(\mathbf{a}\_{ik}^T\mathbf{y} + b_{ik}) \leq 1, ~~, i = 1, \dots, m ~~~~~~(26)\\\
&& \exp(\mathbf{g}_j^T\mathbf{y} + h_j) = 1, ~ j= 1, \dots, p
\end{eqnarray}
\\]

với \\(\mathbf{a}_{ik} \in \mathbb{R}^n, i = 1, \dots, p\\) và \\(\mathbf{g}_i \in \mathbb{R}^n\\).

Với chú ý rằng hàm số \\(\log \sum_{i=1}^m \exp(g_i(\mathbf{x}))\\) là môt hàm _lồi_ nếu \\(g_i\\) là các hàm _lồi_ (tôi xin bỏ qua phần chứng minh), ta có thể viết lại bài toán \\((26)\\) dưới dạng _lồi_ bằng cách lấy \\(\log\\) của các hàm như sau:
<hr>
GP in convex form:
\\[
\begin{eqnarray}
    \text{minimize}\_{\mathbf{y}} \tilde{f}\_0(\mathbf{y}) &=& \log\left\(\sum_{k=1}^{K_0} \exp(\mathbf{a}\_{0k}^T \mathbf{y} + b_{i0})\right\)                          \\\
\text{subject to:}~ \tilde{f}\_i(\mathbf{y}) &=& \log \left\(\sum_{k=1}^{K_i} \exp(\mathbf{a}\_{ik}^T \mathbf{y} + b\_{ik})\right\) \leq 0, ~~ i = 1, \dots, m ~~~~ (27)\\\
\tilde{h}_j(\mathbf{y}) &=& \mathbf{g}_j^T\mathbf{y} + h_j = 0,~~ j = 1, \dots, p
\end{eqnarray}
\\]
<hr>

Lúc này, ta có thể nói rằng GP tương đương với một bài toán tối ưu lồi vì hàm mục tiêu và các hàm bất đẳng thức ràng buộc trong \\((27)\\) đều là hàm lồi, đồng thời điều hiện đẳng thức cuối cùng chính là dạng _affine_. Dạng này thường được gọi là _geometric program in convex form_ (để phân biệt nó với định nghĩa của GP).

<a name="-giai-gp-bang-cvxopt"></a>

### 6.4. Giải GP bằng CVXOPT
Chúng ta quay lại ví dụ về Bài toán đóng thùng _không có ràng buộc_ và hàm mục tiêu là \\(f(x, y, z) = 40x^{-1}y^{-1}z^{-1} + 2xy + 2yz + 2zx\\) là một posynomial. Vậy đây là một GP.

Code cho việc tìm _optimal point_ của bài toán này bằng CVXOPT như sau:

```python
from cvxopt import matrix, solvers
from math import log, exp# gp
from numpy import array
import numpy as np

K = [4]
F = matrix([[-1., 1., 1., 0.],
            [-1., 1., 0., 1.],
            [-1., 0., 1., 1.]])
g = matrix([log(40.), log(2.), log(2.), log(2.)])
solvers.options['show_progress'] = False
sol = solvers.gp(K, F, g)

print('Solution:')
print(np.exp(np.array(sol['x'])))

print('\nchecking sol^5')
print(np.exp(np.array(sol['x']))**5)
```
```
Solution:
[[ 1.58489319]
 [ 1.58489319]
 [ 1.58489319]]

checking sol^5
[[ 9.9999998]
 [ 9.9999998]
 [ 9.9999998]]
```

Nghiệm thu được chính là \\(x = y = z = \sqrt[5]{10}\\). Bạn đọc được khuyến khích đọc thêm chỉ dẫn của hàm `solvers.gp` để hiểu cách thiết lập bài toán.

<a name="-tom-tat"></a>

## 7. Tóm tắt

* Các bài toán tối ưu xuất hiện rất nhiều trong thực tế, trong đó Tối Ưu Lồi đóng một vai trò quan trọng. Trong bài toán Tối Ưu Lồi, nếu tìm được cực trị thì cực trị đó chính là một điểm _optimal_ của bài toán (nghiệm của bài toán).

* Có nhiều bài toán tối ưu không được viết dưới dạng _convex_ nhưng có thể biến đổi về dạng _convex_, ví dụ như bài toán Geometric Programming.

* Linear Programming và Quadratic Programming đóng một vài trò quan trọng trong toán tối ưu, được sử dụng nhiều trong các thuật toán Machine Learning.

* Thư viện CVXOPT được dùng để tối ưu nhiều bài toán tối ưu lồi, rất dễ sử dụng và thời gian chạy tương đối nhanh.

<a name="-tai-lieu-tham-khao"></a>

## 8. Tài liệu tham khảo
[1] [Convex Optimization](http://stanford.edu/~boyd/cvxbook/) – Boyd and Vandenberghe, Cambridge University Press, 2004.

[2] [CVXOPT](http://cvxopt.org/).
