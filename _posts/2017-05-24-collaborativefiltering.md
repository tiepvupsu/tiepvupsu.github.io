---
layout: post
comments: true
title:  "Bài 24: Neighborhood-Based Collaborative Filtering"
title2:  "24. Neighborhood-Based Collaborative Filtering"
date:   2017-05-24 15:22:00
permalink: 2017/05/24/collaborativefiltering/
mathjax: true
tags: Recommendation-systems
category: Recommendation-systems
sc_project: 11351973
sc_security: 2a35b75e
img: /assets/24_collaborativefiltering/item_cf.png
summary: Phương pháp suy luận mức quan tâm của một người dùng cho một sản phẩn dựa trên hành vi của những người dùng tương tự. 
---

**Trong trang này:**
<!-- MarkdownTOC -->

- [1. Giới thiệu](#-gioi-thieu)
- [2. User-user Collaborative Filtering](#-user-user-collaborative-filtering)
    - [2.1. Similarity functions](#-similarity-functions)
    - [2.2. Rating prediction](#-rating-prediction)
- [3. Item-item Collaborative Filtering](#-item-item-collaborative-filtering)
- [4. Lập trình Collaborative Filtering trên Python](#-lap-trinh-collaborative-filtering-tren-python)
    - [4.1. `class CF`](#-class-cf)
    - [4.2. Áp dụng vào ví dụ](#-ap-dung-vao-vi-du)
    - [4.3. Áp dụng lên MovieLens 100k](#-ap-dung-len-movielens-k)
- [5. Thảo luận](#-thao-luan)
- [6. Tài liệu tham khảo](#-tai-lieu-tham-khao)

<!-- /MarkdownTOC -->


<a name="-gioi-thieu"></a>

## 1. Giới thiệu

Trong [Content-based Recommendation Systems](/2017/05/17/contentbasedrecommendersys/), chúng ta đã làm quen với một Hệ thống gợi ý sản phẩm đơn giản dựa trên đặc trưng của mỗi _item_. Đặc điểm của Content-based Recommendation Systems là việc xây dựng mô hình cho mỗi _user_ không phụ thuộc vào các _users_ khác mà phụ thuộc vào _profile_ của mỗi _items_. Việc làm này có lợi thế là tiết kiệm bộ nhớ và thời gian tính toán. Đồng thời, hệ thống có khả năng _tận dụng_ các thông tin đặc trưng của mỗi _item_ như được mô tả trong _bản mô tả_ (description) của mỗi _item_. _Bản mô tả_ này có thể được xây dựng bởi nhà cung cấp hoặc được thu thập bằng cách yêu cầu _users_ gắn _tags_ cho _items_. Việc xây dựng _feature vector_ cho mỗi _item_ thường bao gồm các kỹ thuật Xử lý ngôn ngữ tự nhiên (Natural Language Processing - NLP). 

Cách làm trên có hai nhược điểm cơ bản. _Thứ nhất_, khi xây dựng mô hình cho một user, các hệ thống Content-based không tận dụng được thông tin từ các _users_ khác. Những thông tin này thường rất hữu ích vì hành vi mua hàng của các _users_ thường được nhóm thành một vài nhóm đơn giản; nếu biết hành vi mua hàng của một vài _users_ trong nhóm, hệ thống nên _suy luận_ ra hành vi của những _users_ còn lại. _Thứ hai_, không phải lúc nào chúng ta cũng có _bản mô tả_ cho mỗi _item_. Việc yêu cầu _users_ gắn _tags_ còn khó khăn hơn vì không phải ai cũng sẵn sàng làm việc đó; hoặc có làm nhưng sẽ mang xu hướng cá nhân. Các thuật toán NLP cũng phức tạp hơn ở việc phải xử lý các từ gần nghĩa, viết tắt, sai chính tả, hoặc được viết ở các ngôn ngữ khác nhau. 

Những nhược điểm phía trên có thể được giải quyết bằng _Collaborative Filtering_ (CF). Trong bài viết này, tôi sẽ trình bày tới các bạn một phương pháp CF có tên là _Neighborhood-based Collaborative Filtering (NBCF)_. Bài tiếp theo sẽ trình bày về một phương pháp CF khác có tên _Matrix Factorization Collaborative Filtering_. Khi chỉ nói Collaborative Filtering, chúng ta sẽ ngầm hiểu rằng phương pháp được sử dụng là _Neighborhood-based_. 

Ý tưởng cơ bản của NBCF là xác định _mức độ quan tâm_ của một _user_ tới một _item_ dựa trên các _users_ khác _gần giống_ với _user_ này. Việc _gần giống nhau_ giữa các _users_ có thể được xác định thông qua _mức độ quan tâm_ của các _users_ này tới các _items_ khác mà hệ thống đã biết.  Ví dụ, _A, B_ đều thích phim _Cảnh sát hình sự_, tức đều _rate_ bộ phim này 5 sao. Ta đã biết _A_ cũng thích _Người phán xử_, vậy nhiều khả năng _B_ cũng thích bộ phim này. 

Các bạn có thể đã hình dung ra, hai câu hỏi quan trọng nhất trong một hệ thống Neighborhood-based Collaborative Filtering là:

* Làm thế nào xác định được _sự giống nhau_ giữa hai _users_?
* Khi đã xác định được các _users_ *gần giống nhau* (*similar users*) rồi, làm thế nào dự đoán được _mức độ quan tâm_ của một _user_ lên một _item_?

Việc xác định mức độ quan tâm của mỗi *user* tới một *item* dựa trên mức độ quan tâm của _similar users_ tới _item_ đó còn được gọi là _User-user collaborative filtering_. Có một hướng tiếp cận khác được cho là làm việc hiệu quả hơn là _Item-item collaborative filtering_. Trong hướng tiếp cận này, thay vì xác định _user similarities_, hệ thống sẽ xác định _item similarities_. Từ đó, hệ thống gợi ý những _items_ _gần giống với_ những _items_ mà user có mức độ quan tâm cao. 

Cấu trúc của bài viết như sau: Mục 2 sẽ trình bày _User-user Collaborative Filtering_. Mục 3 sẽ nêu một số hạn chế của  _User-user Collaborative Filtering_ và cách khắc phục bằng _Item-item Collaborative Filtering_. Kết quả của hai phương pháp này sẽ được trình bày qua ví dụ trên cơ sở dữ liệu [MovieLens 100k](/2017/05/17/contentbasedrecommendersys/#-co-so-du-lieu-movielens-k) trong Mục 4. Một vài thảo luận và Tài liệu tham khảo được cho trong Mục 5 và 6.

<a name="-user-user-collaborative-filtering"></a>

## 2. User-user Collaborative Filtering 

<a name="-similarity-functions"></a>

### 2.1. Similarity functions 

Công việc quan trọng nhất phải làm trước tiên trong User-user Collaborative Filtering là phải xác định được _sự giống nhau_ (_similarity_) giữa hai _users_. Dữ liệu duy nhất chúng ta có là _Utility matrix_ \\(\mathbf{Y}\\), vậy nên _sự giống nhau_ này phải được xác định dựa trên các cột tương ứng với hai _users_ trong ma trận này. Xét ví dụ trong Hình 1. 

<hr>
<div>
<table width = "100%" style = "border: 0px solid white">
    <tr >
        <td width="40%" style = "border: 0px solid white" align = "center">
        <img style="display:block;" width = "100%" src = "\assets\24_collaborativefiltering\utility.png">
         </td>
        <td width="40%" style = "border: 0px solid white" align = "justify">
        Hình 1: Ví dụ về utility matrix dựa trên số sao một <em>user rate</em> cho một <em>item</em>. Một cách trực quan, <em>hành vi</em> của \(u_0\) giống với \(u_1\) hơn là \(u_2, u_3, u_4, u_5, u_6\). Từ đó có thể dự đoán rằng \(u_0\) sẽ quan tâm tới \(i_2\) vì \(u_1\) cũng quan tâm tới <em>item</em> này.
        </td>
    </tr>
</table>
</div>
<hr>

Giả sử có các _users_ từ \\(u_0\\) đến \\(u_6\\) và các _items_ từ \\(i_0\\) đến \\(i_4\\) trong đó các số trong mỗi ô vuông thể hiện _số sao_ mà mỗi _user_ đã _rated_ cho _item_ với giá trị cao hơn thể hiện *mức độ quan tâm* cao hơn. Các dấu hỏi chấm là các giá trị mà hệ thống cần phải đi tìm. Đặt _mức độ giống nhau_ của hai _users_ \\(u_i, u_j\\) là \\(\text{sim}(u_i, u_j)\\). 

Quan sát đầu tiên chúng ta có thể nhận thấy là các \\(u_0, u_1\\) _thích_ \\(i_0, i_1, i_2\\) và _không thích_ \\(i_3, i_4\\) cho lắm. Điều ngược lại xảy ra ở các _users_ còn lại. Vì vậy, một _similiarity function_ tốt cần đảm bảo:

 \\[\text{sim}(u_0, u_1) > \text{sim}(u_0, u_i), ~\forall i > 1.\\] 

Từ đó, để xác định _mức độ quan tâm_ của \\(u_0\\) lên \\(i_2\\), chúng ta nên dựa trên _hành vi_ của \\(u\_1\\) lên sản phẩm này. Rất may rằng \\(u_1\\) đã _thích_ \\(i_2\\) nên hệ thống cần _recommend_ \\(i\_2\\) cho \\(u_0\\).

Câu hỏi đặt ra là: hàm số _similarity_ nào là tốt? Để đo _similarity_ giữa hai _users_, cách thường làm là xây dựng _feature vector_ cho mỗi _user_ rồi áp dụng một hàm có khả năng đo _similarity_ giữa hai vectors đó. Chú ý rằng việc xây dựng feature vector này khác với việc xây dựng [item profiles](2017/05/17/contentbasedrecommendersys/#-item-profiles) như trong Content-based Recommendation Systems. Các vectors này được xây dựng trực tiếp dựa trên Utility matrix chứ không dùng dữ liệu ngoài như item profiles. Với mỗi user, thông tin duy nhất chúng ta biết là các _ratings_ mà _user_ đó đã thực hiện, tức cột tương ứng với _user_ đó trong Utility matrix. Tuy nhiên, khó khăn là các cột này thường có rất nhiều _mising ratings_ vì mỗi _user_ thường chỉ _rated_ một số lượng rất nhỏ các _items_. Cách khắc phục là bằng cách nào đó, ta _giúp_ hệ thống _điền_ các giá trị này sao cho việc điền không làm ảnh hưởng nhiều tới _sự giống nhau_ giữa hai vector. Việc _điền_ này chỉ phục vụ cho việc tính _similarity_ chứ không phải là _suy luận_ ra giá trị cuối cùng. 

Vậy mỗi dấu '?' nên được thay bởi giá trị nào để hạn chế việc sai lệch quá nhiều? Một lựa chọn bạn có thể nghĩ tới là thay các dấu '?' bằng giá trị '0'. Điều này không thực sự tốt vì giá trị '0' tương ứng với mức độ quan tâm thấp nhất. Một giá trị _an toàn_ hơn là 2.5 vì nó là trung bình cộng của 0, mức thấp nhất, và 5, mức cao nhất. Tuy nhiên, giá trị này có hạn chế đối với những _users_ _dễ tính_ hoặc _khó tính_. Với các _users_ dễ tính, _thích_ tương ứng với 5 sao, _không thích_ có thể ít sao hơn 1 chút, 3 sao chẳng hạn. Việc chọn giá trị 2.5 sẽ khiến cho các _items_ còn lại là quá _negative_ đối với _user_ đó. Điều ngược lại xảy ra với những _user_ khó tính hơn khi chỉ cho 3 sao cho các _items_ họ thích và ít sao hơn cho những _items_ họ không thích. 

Một giá trị khả dĩ hơn cho việc này là trung bình cộng của các _ratings_ mà _user_ tương ứng đã thực hiện. Việc này sẽ tránh được việc _users_ quá khó tính hoặc dễ tính, tức lúc nào cũng có những _items_ mà một _user_ thích hơn so với những _items_ khác. 

Hãy cùng xem ví dụ trong Hình 2a) và 2b).

<hr>
<div class="imgcap">
<img src ="\assets\24_collaborativefiltering\user_cf.png" align = "center" width = "800">
<div class = "thecap" align = "left">Hình 2: Ví dụ mô tả User-user Collaborative Filtering. a) Utility Matrix ban đầu. b) Utility Matrix đã được chuẩn hoá. c) User similarity matrix. d) Dự đoán các (normalized) <em>ratings</em> còn thiếu. e) Ví dụ về cách dự đoán normalized rating của \(u_1\) cho \(i_1\). f) Dự đoán các (denormalized) <em>ratings</em> còn thiếu. </div>
</div> 
<hr>

**Chuẩn hoá dữ liệu:**

Hàng cuối cùng trong Hình 2a) là giá trị trung bình của _ratings_ cho mỗi _user_. Giá trị cao tương ứng với các _user dễ tính_ và ngược lại. Khi đó, nếu tiếp tục trừ từ mỗi _rating_ đi giá trị này và thay các giá trị chưa biết bằng 0, ta sẽ được _normalized utility matrix_ như trong Hình 2b). Bạn có thể thắc mắc tại sao bước chuẩn hoá này lại quan trọng, câu trả lời ở ngay đây: 

* Việc trừ đi trung bình cộng của mỗi _cột_ khiến trong trong mỗi cột có những giá trị dương và âm. Những giá trị dương tương ứng với việc _user thích item_, những giá trị âm tương ứng với việc _user không thích item_. Những giá trị bằng 0 tương ứng với việc _chưa xác định_ được liệu _user_ có thích _item_ hay không. 
* Về mặt kỹ thuật, số chiều của utility matrix là rất lớn với hàng triệu _users_ và _items_, nếu lưu toàn bộ các giá trị này trong một ma trận thì khả năng cao là sẽ không đủ bộ nhớ. Quan sát thấy rằng vì số lượng _ratings_ biết trước thường là một số rất nhỏ so với kích thước của utility matrix, sẽ tốt hơn nếu chúng ta lưu ma trận này dưới dạng _sparse matrix_, tức chỉ lưu các giá trị khác không và vị trí của chúng. Vì vậy, tốt hơn hết, các dấu '?' nên được thay bằng giá trị '0', tức chưa xác định liệu _user_ có thích _item_ hay không. Việc này không những tối ưu bộ nhớ mà việc tính toán _similarity matrix_ sau này cũng hiệu quả hơn. 

Sau khi đã chuẩn hoá dữ liệu như trên, một vài _similiraty function_ thường được sử dụng là: 

**Cosine Similarity:**

Đây là hàm được sử dụng nhiều nhất, và cũng quen thuộc với các bạn nhất. Nếu các bạn không nhớ công thức tính \\(\text{cos}\\) của góc giữa hai vector \\(\mathbf{u}_1, \mathbf{u}_2\\) trong chương trình phổ thông, thì dưới đây là công thức:

\\[
\text{cosine_similarity}(\mathbf{u}_1, \mathbf{u}_2) =\text{cos}(\mathbf{u}_1, \mathbf{u}_2) 
=  \frac{\mathbf{u}_1^T\mathbf{u}_2}{ \|\|\mathbf{u}_1\|\|_2.\|\|\mathbf{u}_2\|\|_2}~~~~ (1)
\\]

Trong đó \\(\mathbf{u}\_{1, 2}\\) là vectors tương ứng với _users 1, 2_  **đã được chuẩn hoá** như ở trên.

Có một tin vui là python có hàm hỗ trợ tính toán hàm số này một cách hiệu quả. 

Độ _similarity_ của hai vector là 1 số trong đoạn [-1, 1]. Giá trị bằng 1 thể hiện hai vector hoàn toàn _similar_ nhau. Hàm số \\(\text{cos}\\) của một góc bằng 1 nghĩa là góc giữa hai vector bằng 0, tức một vector bằng tích của một số dương với vector còn lại. Giá trị \\(\text{cos}\\) bằng -1 thể hiện hai vector này hoàn toàn trái ngược nhau. Điều này cũng hợp lý , tức khi _hành vi_ của hai _users_ là hoàn toàn ngược nhau thi _similarity_ giữa hai vector đó là thấp nhất. 

Ví dụ về cosine_similarity của các _users_ trong Hình 2b) được cho trong Hình 2c). Similarity matrix \\(\mathbf{S}\\) là một ma trận đối xứng vì \\(\text{cos}\\) là một hàm chẵn, và nếu _user A giống user B_ thì điều ngược lại cũng đúng. Các ô màu xanh trên đường chéo đều bằng 1 vì đó là \\(\text{cos}\\) của góc giữa 1 vector và chính nó, tức \\(\text{cos}(0) = 1\\). Khi tính toán ở các bước sau, chúng ta không cần quan tâm tới các giá trị 1 này. Tiếp tục quan sát các vector hàng tương ứng với \\(u_0, u_1, u_2\\), chúng ta sẽ thấy một vài điều thú vị:

* \\(u\_0\\) *gần* với \\(u_1\\) và \\(u_5\\) (độ giống nhau là dương) hơn các _users_ còn lại. Việc _similarity_ cao giữa \\(u\_0\\) và \\(u_1\\) là dễ hiểu vì cả hai đều có xu hướng quan tâm tới \\(i_0, i_1, i_2\\) hơn các _items_ còn lại. Việc \\(u\_0\\) _gần_ với \\(u_5\\) thoạt đầu có vẻ vô lý vì \\(u_5\\) đánh giá thấp các _items_ mà \\(u\_0\\) đánh giá cao (Hình 2a)); tuy nhiên khi nhìn vào ma trận utility đã chuẩn hoá ở Hình 2b), ta thấy rằng điều này là hợp lý. Vì _item_ duy nhất mà cả hai _users_ này đã cung cấp thông tin là \\(i_1\\) với các giá trị tương ứng đều là _tích cực_. 

* \\(u\_1\\) gần với \\(u_0\\) và xa các _users_ còn lại. 

* \\(u\_2\\) gần với \\(u_3, u_4, u_5, u_6\\) và xa các _users_ còn lại. 

Từ _similarity matrix_ này, chúng ta có thể phân nhóm các _users_ ra làm hai nhóm \\((u\_0, u_1)\\) và \\((u_2, u_3, u_4, u_5, u_6)\\). Vì ma trận \\(\mathbf{S}\\) này nhỏ nên chúng ta có thể dễ dàng quan sát thấy điều này; khi số _users_ lớn hơn, việc xác định bằng _mắt thường_ là không khả thi. Việc xây dựng thuật toán phân nhóm các _users_ (_users clustering_) _rất có thể_ sẽ được trình bày ở một trong các bài viết tiếp theo. 

Có một chú ý quan trọng ở đây là khi số lượng _users_ lớn, ma trận \\(\mathbf{S}\\) cũng rất lớn và nhiều khả năng là không có đủ bộ nhớ để lưu trữ, ngay cả khi chỉ lưu hơn một nửa số các phần tử của ma trận _đối xứng_ này. Với các trường hợp đó, mới mỗi _user_, chúng ta chỉ cần tính và lưu kết quả của một hàng của _similarity matrix_, tương ứng với việc độ _giống nhau_ giữa _user_ đó và các _users_ còn lại. 

Trong bài viết này, tôi sẽ sử dụng _similarity function_ này. 

**Person corelation:** 

Tôi xin không đi chi tiết về phần này, bạn đọc quan tâm có thể đọc thêm [Pearson correlation coefficient - Wikipedia](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)

<a name="-rating-prediction"></a>

### 2.2. Rating prediction   

Việc xác định mức độ quan tâm của một _user_ lên một _item_ dựa trên các _users gần nhất_ (_neighbor users_) này rất giống với những gì chúng ta thấy trong [Bài 6: K-nearest neighbors](/2017/01/08/knn/). Khi làm việc với _large-scale problems_, chúng ta sẽ thấy thêm rằng phương pháp _lười học_ K-nearest neighbors (KNN) được sử dụng rất nhiều vì tính đơn giản của nó. Tất nhiên, chúng ta không thể trực tiếp sử dụng KNN mà còn cần phải làm thêm nhiều bước trung gian nữa. 

Tương tự như KNN, trong Collaborative Filtering, _missing rating_ cũng được xác định dựa trên thông tin về \\(k\\) _neighbor users_. Tất nhiên, chúng ta chỉ quan tâm tới **các _users_ đã _rated item_ đang xét**. _Predicted rating_ thường được xác định là _trung bình có trọng số_ của các _ratings_ **đã chuẩn hoá**. Có một điểm cần lưu ý, trong KNN, các trọng số được xác định dựa trên _distance_ giữa 2 điểm, và các _distance_ này là các số không âm. Trong khi đó, trong CF, các trọng số được xác định dựa trên _similarity_ giữa hai _users_, những trọng số này có thể nhỏ hơn 0 như trong Hình 2c).

Công thức phổ biến được sử dụng để dự đoán _rating_ của \\(u\\) cho \\(i\\) là:

\\[
\hat{y}\_{i, u} = \frac{\sum_{u_j \in \mathcal{N}(u, i)} \bar{y}\_{i, u_j} \text{sim}(u, u_j)}{\sum_{u_j \in \mathcal{N}(u, i)} |\text{sim}(u, u_j)|} ~~~~ (2)
\\]
(sự khác biết so với trung bình có trọng số là mẫu số có sử dụng trị tuyệt đối để xử lý các số âm).

trong đó \\(\mathcal{N}(u, i)\\) là tập hợp \\(k\\) _users_ trong _neighborhood_  (tức có _similarity_ cao nhất) của \\(u\\) mà **đã rated** \\(i\\).

Hình 2d) thể hiện việc _điền_ các giá trị còn thiếu trong _normalized utility matrix_. Các ô màu nền đỏ thể hiện các giá trị dương, tức các _items_ mà _có thể users_ đó _quan tâm_. Ở đây, tôi đã lấy ngưỡng bằng 0, chúng ta hoàn toàn có thể chọn các _ngưỡng_ khác 0. 

Một ví dụ về việc tính _normalized rating_ của \\(u_1\\) cho \\(i_1\\) được cho trong Hình 2e) với số _nearest neighbors_ là \\(k = 2\\). Các bước thực hiện là: 

1. Xác định các _users_ đã _rated_ \\(i_1\\), đó là \\(u_0, u_3, u_5\\).

2. Xác định _similarities_ của \\(u_1\\) với các _users_ này ta nhận được \\(\{0.83, -0.40, -0.23\}\\). Hai (\\(k = 2\\)) giá trị lớn nhất là \\(0.83\\) và \\(-0.23\\) tương ứng với \\(u_0\\) và \\(u_5\\).

3. Xác định các _normalized ratings_ của \\(u_0, u_5\\) cho \\(i_1\\), ta thu được hai giá trị lần lượt là \\(0.75\\) và \\(0.5\\). 

4. Dự đoán kết quả: 

\\[
\hat{y}_{i_1, u_1} = \frac{0.83\times 0.75 + (-0.23)\times 0.5}{0.83 + |-0.23|} \approx 0.48
\\]

Việc quy đổi các giá trị ratings đã chuẩn hoá về thang 5 có thể được thực hiện bằng cách cộng các cột của ma trận \\(\hat{\mathbf{Y}}\\) với giá trị rating trung bình của mỗi _user_ như đã tính trong Hình 2a). 

Việc hệ thống quyết định _recommend items_ nào cho mỗi _user_ có thể được xác định bằng nhiều cách khác nhau. Có thể sắp xếp _unrated items_ theo thứ tự tự lớn đến bé của các _predicted ratings_, hoặc chỉ chọn các _items_ có _normalized predicted ratings_ dương - tương ứng với việc _user_ này có nhiều khả năng thích hơn. 


Trước khi vào phần lập trình cho User-user CF, chúng ta cùng xem xét Item-item CF. 

<a name="-item-item-collaborative-filtering"></a>

## 3. Item-item Collaborative Filtering 
Một số hạn chês của User-user CF:

* Trên thực tế, số lượng _users_ luôn lớn hơn số lượng _items_ rất nhiều. Kéo theo đó là _Similarity matrix_ là rất lớn với số phần tử phải lưu giữ là hơn 1 nửa của bình phương số lượng _users_ (chú ý rằng ma trận này là đối xứng). Việc này, như đã đề cập ở trên, khiến cho việc lưu trữ ma trận này trong nhiều trường hợp là không khả thi. 

* Ma trận Utility \\(\mathbf{Y}\\) thường là rất _sparse_. Với số lượng _users_ rất lớn so với số lượng _items_, rất nhiều cột của ma trận này sẽ rất _sparse_, tức chỉ có một vài phần tử khác 0. Lý do là _users_ thường _lười_ rating. Cũng chính vì việc này, một khi _user_ đó thay đổi _rating_ hoặc rate thêm _items_, trung bình cộng các _ratings_ cũng như vector chuẩn hoá tương ứng với _user_ này thay đổi nhiều. Kéo theo đó, việc tính toán ma trận Similarity, vốn tốn nhiều bộ nhớ và thời gian, cũng cần được thực hiện lại. 

Ngược lại, nếu chúng ta tính toán _similarity_ giữa các _items_ rồi _recommend_ những _items_ gần _giống_ với _item_ yêu thích của một _user_ thì sẽ có những lợi ích sau: 

* Vì số lượng _items_ thường nhỏ hơn số lượng _users_, Similarity matrix trong trường hợp này cũng nhỏ hơn nhiều, thuận lợi cho việc lưu trữ và tính toán ở các bước sau. 

* Vì số lượng phần tử đã biết trong Utility matrix là như nhau nhưng số hàng (_items_) ít hơn số cột (_users_), nên trung bình, mỗi hàng của ma trận này sẽ có nhiều phần tử đã biết hơn số phần tử đã biết trong mỗi cột. Việc này cũng dễ hiểu vì mỗi _item_ có thể được _rated_ bởi nhiều _users_. Kéo theo đó, giá trị trung bình của mỗi hàng ít bị thay đổi hơn khi có thêm một vài _ratings_. Như vậy, việc cập nhật ma trận Similarity Matrix có thể được thực hiện ít thường xuyên hơn. 

Cách tiếp cận thứ hai này được gọi là _Item-item Collaborative Filtering_. Hướng tiếp cận này được sử dụng nhiều trong thực tế hơn. 

Quy trình dự đoán _missing ratings_ cũng tương tự như trong User-user CF. Hình 3 mô tả quy trình này với ví dụ nêu ở phần trên. 

<div class="imgcap">
<img src ="\assets\24_collaborativefiltering\item_cf.png" align = "center" width = "800">
<div class = "thecap" align = "left">Hình 3: Ví dụ mô tả Item-Item Collaborative Filtering. a) Utility Matrix ban đầu. b) Utility Matrix đã được chuẩn hoá. c) User similarity matrix. d) Dự đoán các (normalized) <em>ratings</em> còn thiếu.</div>
</div> 

Có một điểm thú vị trong Similarity matrix ở Hình 3c) là có các phần tử trong hai hình vuông xanh và đỏ đều là các số không âm, các phần tử bên ngoài là các số âm. Việc này thể hiện rằng các _items_ có thể được chia thành 2 nhóm rõ rệt với những _items_ có _similarity_ không âm vào 1 nhóm. Như vậy, một cách _vô tình_, chúng ta đã thực hiện việc _item clustering_. Việc này sẽ giúp ích rất nhiều trong việc dự đoán ở phần sau. 

Kết quả về việc chọn _items_ nào để _recommend_ cho mỗi _user_ được thể hiện bởi các ô màu đỏ trong Hình 3d). Kết quả này có khác một chút so với kết quả tìm được bởi User-user CF ở 2 cột cuối cùng tương ứng với \\(u_5, u_6\\). Dường như kết quả này _hợp lý_ hơn vì từ Utility Matrix, có hai nhóm _users_ thích hai nhóm _items_ khác nhau. (Bạn có nhận ra không?)

**Về mặt tính toán, Item-item CF có thể nhận được từ User-user CF bằng cách chuyển vị (transpose) ma trận utility, và coi như _items_ đang _rate_ _users_. Sau khi tính ra kết quả cuối cùng, ta lại chuyển vị một lần nữa để thu được kết quả.**

Phần 3 dưới đây sẽ mô tả cách lập trình cho Collaborative Filtering trên python. Chú ý rằng thư viện `sklearn` tôi vẫn dùng không có các modules cho Recommendation Systems.    

<a name="-lap-trinh-collaborative-filtering-tren-python"></a>

## 4. Lập trình Collaborative Filtering trên Python 
Trong bày này, tôi lập trình theo hướng Hướng Đối Tượng cho class CF. `Class` này được sử dụng chung cho cả User-user và Item-item CF. Trước hết, chúng ta sẽ thử nghiệm với ví dụ nhỏ trong bài, sau đó sẽ áp dụng vào bài toán với cơ sở dữ liệu MovieLens. 

Dưới đây là file [`ex.dat`](https://github.com/tiepvupsu/tiepvupsu.github.io/blob/master/assets/24_collaborativefiltering/python/ex.dat) mô tả dữ liệu đã biết cho ví dụ. Thứ tự của ba cột là `user_id`, `item_id`, và `rating`. Ví dụ, hàng đầu tiên nghĩa là `u_0` rates `i_0` số sao là `5`.

Khi làm việc với Item-item CF, chúng ta chỉ cần đổi vị trí của hai cột đầu tiên để nhận được ma trận chuyển vị. 

```
0 0 5.
0 1 4.
0 3 2.
0 4 2.
1 0 5.
1 2 4.
1 3 2.
1 4 0.
2 0 2.
2 2 1.
2 3 3.
2 4 4.
3 0 0.
3 1 0.
3 3 4.
4 0 1.
4 3 4.
5 1 2.
5 2 1.
6 2 1.
6 3 4.
6 4 5.
```

<a name="-class-cf"></a>

### 4.1. `class CF`


**Khởi tạo `class CF`** 

Dữ liệu đầu vào của hàm khởi tạo `class CF` là ma trận Utility `Y_data` được lưu dưới dạng một ma trận với 3 cột, `k` là số lượng các điểm lân cận được sử dụng để dự đoán kết quả. `dist_func` là hàm đó _similarity_ giữa hai vectors, mặc định là `cosine_similarity` được lấy từ `sklearn.metrics.pairwise`. Bạn đọc cũng có thể thử với các giá trị `k` và hàm `dist_func` khác nhau. Biến `uuCF` thể hiện việc đang sử dụng User-user CF (1) hay Item-item CF(0). 

```python
import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse 

class CF(object):
    """docstring for CF"""
    def __init__(self, Y_data, k, dist_func = cosine_similarity, uuCF = 1):
        self.uuCF = uuCF # user-user (1) or item-item (0) CF
        self.Y_data = Y_data if uuCF else Y_data[:, [1, 0, 2]]
        self.k = k # number of neighbor points
        self.dist_func = dist_func
        self.Ybar_data = None
        # number of users and items. Remember to add 1 since id starts from 0
        self.n_users = int(np.max(self.Y_data[:, 0])) + 1 
        self.n_items = int(np.max(self.Y_data[:, 1])) + 1
```

**Khi có dữ liệu mới, cập nhận Utility matrix** bằng cách thêm các hàng này vào cuối Utility Matrix. Để cho đơn giản, giả sử rằng không có _users_ hay _items_ mới, cũng không có _ratings_ nào bị thay đổi. 

```python    
    def add(self, new_data):
        """
        Update Y_data matrix when new ratings come.
        For simplicity, suppose that there is no new user or item.
        """
        self.Y_data = np.concatenate((self.Y_data, new_data), axis = 0)
```

**Tính toán normalized utility matrix và Similarity matrix**

```python
    def normalize_Y(self):
        users = self.Y_data[:, 0] # all users - first col of the Y_data
        self.Ybar_data = self.Y_data.copy()
        self.mu = np.zeros((self.n_users,))
        for n in xrange(self.n_users):
            # row indices of rating done by user n
            # since indices need to be integers, we need to convert
            ids = np.where(users == n)[0].astype(np.int32)
            # indices of all ratings associated with user n
            item_ids = self.Y_data[ids, 1] 
            # and the corresponding ratings 
            ratings = self.Y_data[ids, 2]
            # take mean
            m = np.mean(ratings) 
            if np.isnan(m):
                m = 0 # to avoid empty array and nan value
            # normalize
            self.Ybar_data[ids, 2] = ratings - self.mu[n]

        ################################################
        # form the rating matrix as a sparse matrix. Sparsity is important 
        # for both memory and computing efficiency. For example, if #user = 1M, 
        # #item = 100k, then shape of the rating matrix would be (100k, 1M), 
        # you may not have enough memory to store this. Then, instead, we store 
        # nonzeros only, and, of course, their locations.
        self.Ybar = sparse.coo_matrix((self.Ybar_data[:, 2],
            (self.Ybar_data[:, 1], self.Ybar_data[:, 0])), (self.n_items, self.n_users))
        self.Ybar = self.Ybar.tocsr()

    def similarity(self):
        self.S = self.dist_func(self.Ybar.T, self.Ybar.T)
```

**Thực hiện lại 2 hàm phía trên khi có thêm dữ liệu**. 

```python
    def refresh(self):
        """
        Normalize data and calculate similarity matrix again (after
        some few ratings added)
        """
        self.normalize_Y()
        self.similarity() 
        
    def fit(self):
        self.refresh()
```

**Dự đoán kết quả:**

Hàm `__pred` là hàm dự đoán _rating_ mà _user_ `u` cho _item_ `i` cho trường hợp User-user CF. Vì trong trường hợp Item-item CF, chúng ta cần hiểu ngược lại nên hàm `pred` sẽ thực hiện đổi vị trí hai biến của `__pred`. Để cho API được đơn giản, tôi cho `__pred` là một phương thức private, chỉ được gọi trong `class CF`; `pred` là một phương thức public, thứ tự của biến đầu vào luôn là (_user_, _item_), bất kể phương pháp sử dụng là User-user CF hay Item-item CF. 

```python        
    def __pred(self, u, i, normalized = 1):
        """ 
        predict the rating of user u for item i (normalized)
        if you need the un
        """
        # Step 1: find all users who rated i
        ids = np.where(self.Y_data[:, 1] == i)[0].astype(np.int32)
        # Step 2: 
        users_rated_i = (self.Y_data[ids, 0]).astype(np.int32)
        # Step 3: find similarity btw the current user and others 
        # who already rated i
        sim = self.S[u, users_rated_i]
        # Step 4: find the k most similarity users
        a = np.argsort(sim)[-self.k:] 
        # and the corresponding similarity levels
        nearest_s = sim[a]
        # How did each of 'near' users rated item i
        r = self.Ybar[i, users_rated_i[a]]
        if normalized:
            # add a small number, for instance, 1e-8, to avoid dividing by 0
            return (r*nearest_s)[0]/(np.abs(nearest_s).sum() + 1e-8)

        return (r*nearest_s)[0]/(np.abs(nearest_s).sum() + 1e-8) + self.mu[u]
    
    
    def pred(self, u, i, normalized = 1):
        """ 
        predict the rating of user u for item i (normalized)
        if you need the un
        """
        if self.uuCF: return self.__pred(u, i, normalize)
        return self.__pred(i, u, normalize)
```

**Tìm tất cả các _items_** nên được gợi ý cho _user_ `u` trong trường hợp User-user CF, hoặc tìm tất cả các _users_ có khả năng thích _item_ `u` trong trường hợp Item-item CF

```python            
    def recommend(self, u, normalized = 1):
        """
        Determine all items should be recommended for user u. (uuCF =1)
        or all users who might have interest on item u (uuCF = 0)
        The decision is made based on all i such that:
        self.pred(u, i) > 0. Suppose we are considering items which 
        have not been rated by u yet. 
        """
        ids = np.where(self.Y_data[:, 0] == u)[0]
        items_rated_by_u = self.Y_data[ids, 1].tolist()              
        recommended_items = []
        for i in xrange(self.n_items):
            if i not in items_rated_by_u:
                rating = self.__pred(u, i)
                if rating > 0: 
                    recommended_items.append(i)
        
        return recommended_items 
```


In toàn bộ kết quả:

```python
    def print_recommendation(self):
        """
        print all items which should be recommended for each user 
        """
        print 'Recommendation: '
        for u in xrange(self.n_users):
            recommended_items = self.recommend(u)
            if self.uuCF:
                print '    Recommend item(s):', recommended_items, 'to user', u
            else: 
                print '    Recommend item', u, 'to user(s) : ', recommended_items
```


Source code cho class này có thể được tìm thấy [ở đây](https://github.com/tiepvupsu/tiepvupsu.github.io/blob/master/assets/24_collaborativefiltering/python/CF.ipynb).

<a name="-ap-dung-vao-vi-du"></a>

### 4.2. Áp dụng vào ví dụ 
Chúng ta sẽ thử với User-user CF trước: 

```python
# data file 
r_cols = ['user_id', 'item_id', 'rating']
ratings = pd.read_csv('ex.dat', sep = ' ', names = r_cols, encoding='latin-1')
Y_data = ratings.as_matrix()

rs = CF(Y_data, k = 2, uuCF = 1)
rs.fit()

rs.print_recommendation()
```

Kết quả:
```
Recommendation: 
    Recommend item(s): [2] to user 0
    Recommend item(s): [1] to user 1
    Recommend item(s): [] to user 2
    Recommend item(s): [4] to user 3
    Recommend item(s): [4] to user 4
    Recommend item(s): [0, 3, 4] to user 5
    Recommend item(s): [1] to user 6
```

Với Item-item Collaborative Filtering:
```python
rs = CF(Y_data, k = 2, uuCF = 0)
rs.fit()

rs.print_recommendation()
```

Kết quả:
```
Recommendation: 
    Recommend item 0 to user(s) :  []
    Recommend item 1 to user(s) :  [1]
    Recommend item 2 to user(s) :  [0]
    Recommend item 3 to user(s) :  [5]
    Recommend item 4 to user(s) :  [3, 4, 5]
```


<a name="-ap-dung-len-movielens-k"></a>

### 4.3. Áp dụng lên MovieLens 100k
Chúng ta cùng quay lại làm với [cơ sở dữ liệu MoiveLens 100k như trong Content-based Recommendation Systems](/2017/05/17/contentbasedrecommendersys/#-bai-toan-voi-co-so-du-lieu-movielens-k). Nhắc lại rằng kết quả của phương pháp này có trung bình lỗi là 1.2 sao với mỗi _rating_. 

Chúng ta cùng xem kết quả với User-user CF và Item-item CF. 

Trước hết, ta cần load dữ liệu.

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

**Kết quả với User-user CF:**

```python
rs = CF(rate_train, k = 30, uuCF = 1)
rs.fit()

n_tests = rate_test.shape[0]
SE = 0 # squared error
for n in xrange(n_tests):
    pred = rs.pred(rate_test[n, 0], rate_test[n, 1], normalized = 0)
    SE += (pred - rate_test[n, 2])**2 

RMSE = np.sqrt(SE/n_tests)
print 'User-user CF, RMSE =', RMSE
```

```
User-user CF, RMSE = 0.995198110088
```

**Kết quả với Item-item CF:**
```python
rs = CF(rate_train, k = 30, uuCF = 0)
rs.fit()

n_tests = rate_test.shape[0]
SE = 0 # squared error
for n in xrange(n_tests):
    pred = rs.pred(rate_test[n, 0], rate_test[n, 1], normalized = 0)
    SE += (pred - rate_test[n, 2])**2 

RMSE = np.sqrt(SE/n_tests)
print 'Item-item CF, RMSE =', RMSE
```

```
Item-item CF, RMSE = 0.986791213271
```

Từ đó ta nhận thấy Item-item CF cho lỗi nhỏ hơn (0.987) so với User-user CF (0.995) và tốt hơn so với Content-based Recommendation Systems ở bài trước (1.2).

_Các bạn cũng có thể thay _neighborhood size_ `k` bằng các giá trị khác và so sánh kết quả._


<a name="-thao-luan"></a>

## 5. Thảo luận

* Collaborative Filtering là một phương pháp gợi ý sản phẩm với ý tưởng chính dựa trên các hành vi của các _users_ khác (collaborative) cùng trên một _item_ để suy ra mức độ quan tâm (filtering) của một _user_ lên sản phẩm. Việc suy ra này được thực hiện dựa trên Similarity matrix đo độ giống nhau giữa các _users_.

* Để tính được Similarity matrix, trước tiên ta cần chuẩn hoá dữ liệu. Phương pháp phổ biến là _mean offset_, tức trừ các _ratings_ đi giá trị trung bình mà một _user_ đưa ra cho các _items. 

* Similarity function thường được dụng là **Cosine similarity** hoặc **Pearson correlation**.

* User-user CF có một vài hạn chế khi lượng _users_ là lớn. Trong các trường hợp đó, Item-item thường được sử dụng và cho kết quả tốt hơn.

* [Source code](https://github.com/tiepvupsu/tiepvupsu.github.io/blob/master/assets/24_collaborativefiltering/python/CF.ipynb)

<a name="-tai-lieu-tham-khao"></a>

## 6. Tài liệu tham khảo

[1] [Recommendation Systems - Stanford InfoLab](http://infolab.stanford.edu/~ullman/mmds/ch9.pdf)

[2] [Collaborative Filtering - Stanford University](https://www.youtube.com/watch?v=h9gpufJFF-0&t=436s)

[3] [Recommendation systems - Machine Learning - Andrew Ng](https://www.youtube.com/watch?v=saXRzxgFN0o&list=PL_npY1DYXHPT-3dorG7Em6d18P4JRFDvH)

[4] Ekstrand, Michael D., John T. Riedl, and Joseph A. Konstan. "[Collaborative filtering recommender systems.](http://herbrete.vvv.enseirb-matmeca.fr/IR/CF_Recsys_Survey.pdf)" Foundations and Trends® in Human–Computer Interaction 4.2 (2011): 81-173.

