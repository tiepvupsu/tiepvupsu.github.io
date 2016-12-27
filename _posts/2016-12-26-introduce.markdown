---
layout: post
comments: true
title:  "Giới thiệu về Machine Learning"
date:   2016-12-26 15:22:00
mathjax: true
---



Những năm gần đây, AI - Artificial Intelligence (Trí Tuệ Nhân Tạo), và cụ thể hơn là Machine Learning (Học Máy hoặc Máy Học - _Với những từ chuyên ngành, tôi sẽ dùng song song cả tiếng Anh và tiếng Việt, tuy nhiên sẽ ưu tiên tiếng Anh vì thuận tiện hơn trong việc tra cứu_) nổi lên như một bằng chứng của cuộc cách mạng công nghiệp lần thứ tư (1 - động cơ hơi nước, 2 - năng lượng điện, 3 - công nghệ thông tin). Trí Tuệ Nhân Tạo đang len lỏi vào mọi lĩnh vực trong đời sống mà có thể chúng ta không nhận ra. Xe tự hành của Google và Tesla, hệ thống tự tag khuôn mặt trong ảnh của Facebook, trợ lý ảo Siri của Apple, hệ thống gợi ý sản phẩm của Amazon, hệ thống gợi ý phim của Netflix, máy chơi cờ vây AlphaGo của Google DeepMind, ..., chỉ là một vài trong vô vàn những ứng dụng của AI/Machine Learning. (Xem thêm [Jarvis - trợ lý thông minh cho căn nhà của Mark Zuckerberg](https://www.facebook.com/zuck/posts/10103351073024591))

Machine Learning là một tập con của AI. Theo định nghĩa của Wikipedia, _Machine learning is the subfield of computer science that "gives computers the ability to learn without being explicitly programmed"_. Nói đơn giản, Machine Learning là một lĩnh vựa nhỏ của Khoa Học Máy Tính, nó có khả năng tự học hỏi dựa trên dữ liệu đưa vào mà không cần phải được lập trình cụ thể. 

Những năm gần đây, khi mà khả năng tính toán của các máy tính được nâng lên một tầm cao mới và lượng dữ liệu khổng lồ được thu thập bởi các hãng công nghệ lớn, Machine Learning đã tiến thêm một bước tiến dài và một lĩnh vực mới được ra đời gọi là Deep Learning (Học Sâu - _thực sự tôi không muốn dịch từ này ra tiếng Việt_). Deep Learning đã giúp máy tính thực thi những việc tưởng chừng như không thể vào 10 năm trước: phân loại cả ngàn vật thể khác nhau trong các bức ảnh, tự tạo chú thích cho ảnh, bắt chước giọng nói và chữ viết của con người, giao tiếp với con người, hay thậm chí cả sáng tác văn hay âm nhạc (Xem thêm [8 Inspirational Applications of Deep Learning](http://machinelearningmastery.com/inspirational-applications-deep-learning/))

<!-- <div style="display:inline-block">
    <img src="/assets/introduce/aimldl.png">
</div>
 -->

<div class="imgcap">
<div >
    <img src="/assets/introduce/aimldl.png" width = "800">
    <!-- <img src="/assets/rl/mdp.png" height="206"> -->
</div>
<div class="thecap">Mối quan hệ giữa AI, Machine Learning và Deep Learning. <br> (Nguồn: <a href="https://blogs.nvidia.com/blog/2016/07/29/whats-difference-artificial-intelligence-machine-learning-deep-learning-ai/">What’s the Difference Between Artificial Intelligence, Machine Learning, and Deep Learning?</a>)</div>
</div>


## Mục đích viết Blog
Nhu cầu về nhân lực ngành Machine Learning (Deep Learning) ngày một cao, kéo theo đó nhu cầu học Machine Learning trên thế giới và ở Việt Nam ngày một lớn. Cá nhân tôi cũng muốn hệ thống lại kiến thức của mình về lĩnh vực này để chuẩn bị cho tương lai (đây là một trong những mục tiêu của tôi trong năm 2017). Tôi sẽ cố gắng đi từ những thuật toán cơ bản nhất của Machine Learning kèm theo các ví dụ và mã nguồn trong mỗi bài viết. Nếu không bận việc đột xuất, tôi sẽ viết 1 bài/tuần. Đồng thơi, tôi cũng muốn nhận được phản hồi của bạn đọc để qua những thảo luận, chúng ta có thể hiểu rõ hơn và nắm bắt các thuật toán này. 

Khi chuẩn bị các bài viết, tôi sẽ giả định rằng bạn đọc có một chút kiến thức về Đại Số Tuyến Tính (Linear Algebra), Xác Suât Thống Kê (Probability and Statistics) và có kinh nghiệm về lập trình Python. Nếu bạn chưa có nhiều kinh nghiệm về các lĩnh vực này, đừng quá lo lắng vì mỗi bài sẽ chỉ sử dụng một vài kỹ thuật cơ bản. Hãy để lại câu hỏi của bạn ở phần Comment bên dưới, tôi sẽ thảo luận thêm với các bạn.

## Tham khảo thêm

### Tiếng Anh
1. [Machine Learning với thầy Andrew Ng trên Coursera](https://www.coursera.org/learn/machine-learning) (_Khóa học nổi tiếng nhất về Machine Learning_)
2. [Deep Learning by Google trên Udacity](https://www.udacity.com/course/deep-learning--ud730) (_Khóa học nâng cao hơn về Deep Learning với Tensorflow_)
3. [Machine Learning mastery](http://machinelearningmastery.com/) (_Các thuật toán Machine Learning cơ bản_)

### Tiếng Việt 
**Lưu ý**: _Các khóa học này tôi chưa từng tham gia, chỉ đưa ra để các bạn tham khảo._

1. [Machine Learning 1/2017](http://tuanvannguyen.blogspot.com/2016/12/cap-nhat-khoa-hoc-ve-machine-learning.html)
2. [Nhập môn Machine Learning - Cao Thanh Hà](https://techmaster.vn/khoa-hoc/25511/machine-learning-co-ban)(_Tech Master_)
3. [Machine Learning trong Xử Lý Ngôn Ngữ Tự Nhiên - Nhóm Đông Du](http://viet.jnlp.org/kien-thuc-co-ban-ve-xu-ly-ngon-ngu-tu-nhien/machine-learning-trong-nlp)
4. [Machine Learning cho người mới bắt đầu - Ông Xuân Hồng](https://ongxuanhong.wordpress.com/). 
5. [Machine Learning book for Vietnamese](https://ml-book-vn.khanhxnguyen.com/)


