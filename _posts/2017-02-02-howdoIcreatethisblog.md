---
layout: post
comments: true
title:  "Blog và các bài viết được tạo như thế nào"
date:   2017-02-02 15:22:00
permalink: 2017/02/02/howdoIcreatethisblog/
mathjax: true
tags: 
category: 
sc_project: 11240681
sc_security: 2198e980
img: /images/logoTet.png
summary: Bạn muốn tạo một blog tương tự hoặc muốn biết tôi chuẩn bị cho các bài viết như thế nào? 
---

Tôi xin phép khởi động năm mới bằng một bài _ngoại truyện_. Bài này sẽ nói về việc tôi tạo blog này như thế nào, và mỗi bài viết được chuẩn bị ra sao. Blog sẽ quay lại với các nội dung về Machine Learning vào đầu tuần tới, Feb 6-7, 2017. 

**Trong trang này:**
<!-- MarkdownTOC -->

- [1. Blog](#-blog)
    - [Server](#server)
    - [Posts](#posts)
    - [Comments](#comments)
    - [Search box](#search-box)
    - [Editor](#editor)
    - [Table of contents](#table-of-contents)
    - [Figures](#figures)
    - [Programming language](#programming-language)
    - [Main references](#main-references)
- [2. Mỗi bài viết được chuẩn bị như thế nào](#-moi-bai-viet-duoc-chuan-bi-nhu-the-nao)

<!-- /MarkdownTOC -->

<a name="-blog"></a>

## 1. Blog 

<a name="server"></a>

### Server 
Tên miền của blog là `github.io`, các bạn quen với git có lẽ đều biết tới trang [https://github.com/ ](https://github.com/) - là một nơi lưu các projects, có hỗ trợ quản lý version và cũng là một Internet hosting service. Tôi dùng trang này để lưu blog vì việc update bài mới rất thuận tiện, và cũng free nữa. Toàn bộ source code của blog này đều có thể xem [tại đây](https://github.com/tiepvupsu/tiepvupsu.github.io). 

Blog được tạo dựa trên nền tảng [Jekyll](https://jekyllrb.com/): hỗ trợ web tĩnh, cực kỳ đơn giản mà vẫn có thể tích hợp được HTML và CSS. Layout của blog được tạo bằng HTML CSS kết hợp với (một chút) Ruby. Việc tạo blog và ba bài đầu tiên chiếm trọn 4 ngày làm việc của tôi. Lúc đó đang là kỳ nghỉ đông, lại ở nhà một mình nên tôi tập trung làm được. 

Hệ điều hành tôi sử dụng là Linux Ubuntu.

<a name="posts"></a>

### Posts 
Các bài viết được tạo dựa trên ngôn ngữ [Markdown](https://en.wikipedia.org/wiki/Markdown#Example) với cú pháp cực kỳ đơn giản. Vài ví dụ:

* In nghiêng

```
*Machine Learning*
```

Kết quả: *Machine Learning*


* In đậm

```
**cơ bản**
```

Kết quả: **cơ bản**

* Insert link

```
[Machine Learning cơ bản](https://tiepvupsu.github.io/)
```

Kết quả: [Machine Learning cơ bản](https://tiepvupsu.github.io/)

* LaTex 

```
\\( \mathbf{w}^T\mathbf{x} \\)
```

Kết quả: \\( \mathbf{w}^T\mathbf{x} \\).

Đây chỉ là một công thức ngắn, các công thức dài sẽ mất thời gian hơn nhiều. Vì Markdown chỉ hỗ trợ LaTeX một cách tối giản nên gõ các công thức toán chiếm khá nhiều thời gian của tôi. Việc phải thường xuyên chuyển đổi qua lại tiếng Việt, tiếng Anh cũng làm chậm đi nhịp gõ. Một điểm nữa, những bài đầu tiên tôi chưa biết tới kiểu gõ Stelex nên các ký tự `[,], {, }` khi dùng LaTeX hoặc thêm link bị hiển thị là `ư, ơ` rất nhiều. Việc này khá khó chịu, rất may là tôi đã tìm được giải pháp Stelex dựa trên gợi ý của một bạn trên facebook. 

(Xem thêm [Markdown cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet))

<a name="comments"></a>

### Comments
Vì jekyll chỉ tạo ra web tĩnh nên tôi phải tìm cách thêm mục comments dựa trên hãng thứ ba (third-party). Trong blog này, tôi sử dụng [Disqus](https://disqus.com/) (đọc như discuss) - là một công cụ lưu trữ comments khá phổ biến. 

Ngoài ra tôi cũng sử dụng các công cụ track page views and locations như [Google Analytics](https://analytics.google.com). Vì công cụ này không cập nhật thường xuyên (khoảng 1 ngày update 1 lần) nên tôi phải dùng thêm [Statcounter](https://statcounter.com/) nữa. 

<a name="search-box"></a>

### Search box
Thời gian đầu blog chưa có ô Search vì Jekyll không hỗ trợ Search tốt như các platform khác như Wordpress. Sau rất nhiều lần thử nghiệm, tôi quyết định dùng [Google Custom Search Engine](https://cse.google.com/). Nhược điểm của công cụ này là nó không cập nhật liên tục mà phải chờ Google _index_ bài mới thì mới ra kết quả tìm kiếm được. Một nhược điểm nữa là khi search sẽ có thêm quảng cáo. Ví dụ, khi tôi search "logistic" cho "logistic function", kết quả trả về sẽ ra vài công ty làm "logistic"! Hiện tại thì đây là biện pháp tốt nhất mà tôi có thể nghĩ tới. 

<a name="editor"></a>

### Editor

Editors chính tôi sử dụng là [Sublime Text](https://www.sublimetext.com/). Tôi có thể viết HTML, CSS, markdown, LaTeX,... trên chỉ một editor này. Sublime Text hỗ trợ rất nhiều các packages tốt cho nhiều ngôn ngữ. Việc quản lý cả một project lớn, tìm kiếm trong toàn bộ project, qua lại giữa các files cũng rất thuận lợi. 

<a name="table-of-contents"></a>

### Table of contents 

Để tạo mục lục cho mỗi bài viết (hỗ trợ tự động sinh links), tôi cần cài package [MarkdownTOC](https://github.com/jonschlinkert/markdown-toc) cho Sublime Text. Có một  khó khăn tôi phải khắc phục. Lấy một ví dụ, tôi đang viết bài Logistic Regression với đường link `/2017/01/27/logisticregression/` và một mục có tên là "Giới thiệu". Nếu dùng MarkdownTOC để tự động tạo mục lục thì nó sẽ tạo ra đường link có dạng `/2017/01/27/logisticregression/#-giới-thiệu`. Vì github.io không hỗ trợ đường link có ký tự tiếng Việt (hoặc có nhưng tôi chưa biết dùng) nên tôi phải sửa package MarkdownTOC một chút bằng cách bỏ dấu tiếng Việt trong link, để nó tự động tạo link có dạng `/2017/01/27/logisticregression/#-gioi-thieu`. 

Nếu bạn nào muốn thử thì có thể sửa `Preferences/Package Settings/MarkdownTOC/Settings - User` như sau:

```
{
  "default_lowercase_only_ascii": true,
  "default_style": "unordered",
  "id_replacements": {
    "-": " ",
    "" : ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "&lt;","&gt;","&amp;","&apos;","&quot;","&#60;","&#62;","&#38;","&#39;","&#34;","!","#","$","&","'","(",")","*","+",",","/",":",";","=","_","?","@","[","]","`","\"", ".","<",">","{","}","™","®","©"],
    "a": ["à", "á", "ả", "ã", "ạ", "â", "ầ", "ấ", "ẩ", "ẫ", "ậ", "ă", "ằ", "ắ", "ẳ", "ẵ", "ặ", "À", "Á", "Ả", "Ã", "Ạ", "Â", "Ầ", "Ấ", "Ẩ", "Ẫ", "Ậ", "Ă", "Ằ", "Ắ", "Ẳ", "Ẵ", "Ặ"],
    "d": ["đ", "Đ"], 
    "e": ["è", "é", "ẻ", "ẽ", "ẹ", "ê", "ề", "ế", "ể", "ễ", "ệ", "È", "É", "Ẻ", "Ẽ", "Ẹ", "Ê", "Ề", "Ế", "Ể", "Ễ", "Ệ"],
    "i": ["ì", "í", "ỉ", "ĩ", "ị", "Ì", "Í", "Ỉ", "Ĩ", "Ị"],
    "o": ["ò", "ó", "ỏ", "õ", "ọ", "ô", "ồ", "ố", "ổ", "ỗ", "ộ", "ơ", "ờ", "ớ", "ở", "ỡ", "ợ", "Ò", "Ó", "Ỏ", "Õ", "Ọ", "Ô", "Ồ", "Ố", "Ổ", "Ỗ", "Ộ", "Ơ", "Ờ", "Ớ", "Ở", "Ỡ", "Ợ"],
    "u": ["ù", "ú", "ủ", "ũ", "ụ", "ư", "ừ", "ứ", "ử", "ữ", "ự", "Ù", "Ú", "Ủ", "Ũ", "Ụ", "Ư", "Ừ", "Ứ", "Ử", "Ữ", "Ự"],
    "y": ["ỳ", "ý", "ỷ", "ỹ", "ỵ", "Ỳ", "Ý", "Ỷ", "Ỹ", "Ỵ"]
  }
}
```


<a name="figures"></a>

### Figures 
Các hình vẽ cần độ chính xác (và thẩm mỹ) cao được vẽ bằng LaTeX với package [TikZ](http://www.texample.net/tikz/) hoặc Python với package [matplotlib](http://matplotlib.org/). Các hình động đều được vẽ bằng Python với package [matplotlib](http://matplotlib.org/).

<a name="programming-language"></a>

### Programming language 

Trong blog này tôi sử dụng Python (cả 2 và 3). Python miễn phí, có nhiều thư viện tốt cho Machine Learning, và theo tôi thì Python cũng đang dần thay thế Matlab. Xem thêm [Python vs Matlab](http://www.pyzo.org/python_vs_matlab.html). Tôi không sử dụng Matlab cho blog vì nhiều bạn ở Việt Nam không có điều kiện mua Matlab bản quyền. Tôi không ủng hộ việc dùng các phần mềm cracked. 

Các packages thường được dùng là [numpy](http://www.numpy.org/), [scipy](https://www.scipy.org/), [scikit-learn](http://scikit-learn.org/stable/), [matplotlib](http://matplotlib.org/), [python-mnist](https://pypi.python.org/pypi/python-mnist/).

Để có thể cài đặt các package này, cách đơn giản là dùng pip. Xem thêm: [How to install Pip on Ubuntu 16.04](https://www.rosehosting.com/blog/how-to-install-pip-on-ubuntu-16-04/).

<a name="main-references"></a>

### Main references 

Các tài liệu trên mạng internet có rất nhiều, nhưng tôi vẫn ưu tiên đọc sách giấy trước. Tôi có sử dụng một số cuốn sách giấy sau:

1. [Learning from data](https://www.google.com/search?q=Learning+from+data&ie=utf-8&oe=utf-8&aq=t). Cuốn này khá ngắn, là cuốn sách Machine Learning đầu tiên tôi mua ($28) và đọc ở Mỹ sau khi xem bài giảng online của tác giả (video online của tác giả có trong link đính kèm).

2. [Pattern Recognition and Machine Learning](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf) - Christoper M. Bishop. Cuốn này rất cơ bản, tôi phải chờ 2 tuần mới mượn được ở thư viện của trường, và phải trả sau chỉ 3 tháng. 

3. [Computer Vision:  Models, Learning, and Inference](http://www.computervisionmodels.com/). Đây là texbook trong môn Computer Vision tôi học ở Penn State. Mặc dù có bản pdf online, tôi vẫn mua ($70) vì tôi thấy nội dung rất đầy đủ, hình vẽ cũng đẹp và dễ hiểu. Nội dung chính là các phương pháp xác suất cho Computer Vision. 

4. [Khóa học Machine Learning của Andrew Ng](https://www.coursera.org/learn/machine-learning?utm_source=gg&utm_medium=sem&campaignid=685340575&adgroupid=37009789418&device=c&keyword=%2Bmachine%20%2Blearning%20%2Bandrew%20%2Bng&matchtype=b&network=g&devicemodel=&adpostion=1t1&creativeid=161364220743&hide_mobile_promo&gclid=CjwKEAiAq8bEBRDuuOuyspf5oyMSJAAcsEyW7LJ5k2hZ8pVy_tUuqEMZ9LwFc6aPeNYknB0QDIODlRoCARfw_wcB). Khóa này rất cơ bản, free. Tác giả là co-founder của coursera và là một big name trong ngành Machine Learning. Theo tôi thì cách trình bày của tác giả rất cơ bản và dễ hiểu. Các bạn nên dành thời gian theo khóa này nếu muốn bước chân vào ngành Machine Learning.

<a name="-moi-bai-viet-duoc-chuan-bi-nhu-the-nao"></a>

## 2. Mỗi bài viết được chuẩn bị như thế nào

Mỗi bài viết thường được tôi chuẩn bị trong trọn vẹn một ngày, từ sáng tới đêm muộn. Vì tên của blog là "Machine Learning cơ bản" nên tôi cố gắng trình bày một cách chi tiết nhất với nhiều hình vẽ minh họa và code mẫu. Tôi không muốn chỉ hướng các bạn tới cách sử dụng các thư viện có sẵn mà muốn các bạn hiểu được nguyên lý đằng sau mỗi thuật toán. Chính vì vậy, việc viết blog chiếm khá nhiều thời gian của tôi, mong các bạn thông cảm nếu tôi ra bài muộn hơn thường lệ.

Khi viết một bài, tôi luôn nghĩ tới việc bài sau sẽ nói gì, có gì liên quan đến bài này, những điều chưa nên viết ở bài này mà để lại ở bài sau. Sau khi viết xong một bài, tôi luôn nghĩ về việc bài tiếp được giới thiệu như thế nào, có những gì cần lưu ý, ... trong khi đi bộ lên trường/về nhà hoặc trong khi tập thể dục. Nếu có điều nào chưa rõ, tôi phải tìm đọc các tài liệu liên quan trước khi bắt tay vào viết.

Khi bắt đầu viết, tôi thường làm theo thứ tự sau:

1. **Viết code:** tôi cần kiểm tra xem những gì mình hiểu có đúng không, code chạy có như ý muốn không. Việc viết code này thường mất 1-2 giờ, đôi khi mất thời gian hơn.

2. **Suy luận toán học:** sau khi code đã chạy theo ý mình, tôi dùng giấy bút để kiểm tra lại các suy luận toán học trong bài, xem có thể dẫn dắt như thế nào từ những quan sát đơn giản. Tôi hạn chế những kiến thức toán mới trong mỗi bài để các bạn mới không cảm thấy ngợp. Phương châm của tôi luôn là "chậm nhưng chắc", bạn đọc muốn đi nhanh hơn có thể đọc thêm sách và các khóa học online. Khoản này mất thêm 1 giờ nữa. 

3. **Vẽ hình:** khoản này mất nhiều thời gian nhất. Trước khi làm blog hơn 1 tháng trước, tôi chưa bao giờ vẽ hình trên matplotlib của Python, chủ yếu dùng LaTeX với TikZ cho các bài báo khoa học. Tuy nhiên, matplotlib hỗ trợ vẽ các hình động rất tốt nên tôi vừa viết vừa học. Cả hai đều có hình vẽ với độ chính xác và thẩm mỹ cao, đồng nghĩa với việc công sức và thời gian bỏ vào cũng cao hơn. Tôi cũng rất thích vẽ hình minh họa nên dành khoảng hơn 2 giờ cho việc vẽ các hình trong mỗi bài. Tôi không thực sự muốn sử dụng các hình có sẵn online vì:

    * Độ phân giải ảnh có thể không cao.

    * Các màu sắc, ký hiệu không thống nhất với các ký hiệu tôi dùng trong bài.

    * Khó có thể chỉnh sửa nếu tôi muốn thêm bớt.

    * Có thể gây khó hiểu với những bạn mới bắt đầu học Machine Learning. 

4. **Viết lần 1:** phần này khá dài nhưng lại không mất nhiều thời gian bằng các phần phía trên vì chủ yếu là gõ theo dòng suy nghĩ đã chuẩn bị trước trong đầu. Vì các bài được viết bằng markdown nên việc chỉnh sửa format không mất thời gian của tôi lắm. Việc mất thời gian nhất là gõ các công thức toán học trong markdown. Không như LaTeX thuần hỗ trợ nhiều packages và có thể define các tên dài, LaTeX trong markdown bị hạn chế rất nhiều (thậm chí không báo lỗi, nhiều khi tôi mất rất nhiều thời gian để tìm lỗi LaTeX không hiển thị đúng). Đôi khi tôi cũng bị nhầm lẫn khi chuyển đổi tiếng Anh/tiếng Việt. Việc này chiếm khoảng 2 giờ nữa. 

5. **Rà soát lại:** tôi thường đọc lại hai lần để sửa các lỗi chính tả, ngữ pháp và xem có ý nào cần thêm bớt không. Tôi dành khoảng 20-30 phút cho phần này. 

6. **Upload** bài lên blog, post bài lên facebook page, facebook cá nhân. Lúc này thường khá muộn, khoảng 1-2 giờ đêm, tôi đi ngủ, sáng hôm sau dậy đếm likes và xem comments. Cũng vào StatCounter xem có bao nhiêu người đã vào blog.

Thi thoảng tôi lại nghĩ ra gì đó mới cho blog thì thường dành một tối nữa để code. Ví dụ như việc chuyển sang giao diện Tết với nền đỏ chữ vàng và hoa đào hoa mai. Màu ưa thích của tôi là màu xanh đậm. 

**Bạn muốn biết thêm điều gì, hãy để lại comment, tôi sẽ cập nhật thêm.** 




