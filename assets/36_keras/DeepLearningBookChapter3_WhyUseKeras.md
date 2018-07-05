
# <p align="center"> Chương 2 Tại sao nên dùng Keras?</p>
Dựa trên https://keras.io/why-use-keras/

## 2.1 Keras ưu tiên  trải nghiệm của người lập trình

Keras tuân theo các phương pháp tốt nhất để thân thiện với người lập trình: cung cấp các API nhất quán và đơn giản, giảm thiểu số lượng thao tác lập trình cần thiết cho các trường hợp sử dụng phổ biến và cung cấp phản hồi rõ ràng khi người dùng gặp lỗi.

Điều này làm cho Keras dễ học và dễ sử dụng. Khi sử dụng Keras, bạn có năng suất cao hơn, cho phép bạn thử nhiều ý tưởng hơn so với đối thủ cạnh tranh của bạn, điều này sẽ giúp bạn giành chiến thắng trong các "cuộc đua" liên quan đến machine learning.

Tính dễ sử dụng này không vì thế mà làm giảm tính linh hoạt: vì Keras tích hợp với các ngôn ngữ học sâu hơn (đặc biệt là TensorFlow). Keras cho phép bạn thực hiện bất cứ điều gì bạn có thể đã xây dựng bằng TensorFlow.

## 2.2 Keras đã được sử dụng rộng rãi trong doanh nghiệp và cộng đồng nghiên cứu

Với hơn 200.000 người dùng cá nhân kể từ tháng 11 năm 2017, Keras đã được áp dụng rất nhiều trong cả doanh nghiệp và cộng đồng nghiên cứu hơn so với bất kỳ frame work nào khác ngoại trừ TensorFlow (và Keras thường được sử dụng kết hợp với TensorFlow).

Hàng ngày trực tiếp hay gián tiếp bạn đều tương tác với các tính năng được xây dựng trên nền Keras - vì Keras đang được sử dụng tại Netflix, Uber, Yelp, Instacart, Zocdoc, Square và nhiều ứng dụng khác. Keras đặc biệt phổ biến trong các công ty khởi nghiệp nơi machine learning và deep learning là cốt lõi trong các sản phẩm của họ.

Keras cũng đặc biệt được yêu thích bởi cộng đồng deep learning, đứng thứ 2 tính về lượng đề cập trong các bài báo khoa học được tải lên máy chủ arXiv.org.

<img src=https://keras.io/img/arxiv-mentions.png width="450" height="300" />
Keras cũng đã được các nhà nghiên cứu sử dụng tại các tổ chức khoa học lớn, đặc biệt là CERN và NASA.

## 2.3 Keras giúp dễ dàng biến các thiết kế thành sản phẩm

Các mô hình Keras của bạn có thể được triển khai một cách dễ dàng trên nhiều nền tảng hơn bất kỳ các deep learning framework nào khác:

+ Trên iOS, thông qua CoreML của Apple (Keras hỗ trợ cho Apple một cách chính thức).
+ Trên Android, thông qua Android TensorFlow runtime.
+ Trong trình duyệt, thông qua các JavaScript runtimes được tăng tốc bởi GPU như Keras.js và WebDNN.
+ Trên Google Cloud, thông qua TensorFlow-Serving.
+ Trong một chương trình phụ trợ webapp Python (chẳng hạn như một ứng dụng Flask).
+ Trên JVM, thông qua mô hình DL4J do SkyMind cung cấp.
+ Trên Raspberry Pi.

## 2.4 Keras hỗ trợ đa backend engines và không giới hạn bạn vào một hệ sinh thái

Các mô hình Keras của bạn có thể được phát triển với các deep learning backends khác nhau. Bạn có thể huấn luyện mô hình bằng một backend này và tải mô hình đó bằng một backend khác. Các backends sẵn có bao gồm:
+ TensorFlow (từ Google)
+ CNTK (từ Microsoft)
+ Theano

Amazon hiện cũng đang làm việc để phát triển MXNet backend cho Keras.
Mô hình Keras của bạn có thể được huấn luyện trên một số nền tảng phần cứng khác nhau ngoài CPU:

+ NVIDIA GPU
+ Google TPUs, thông qua TensorFlow backend và Google Cloud
+ Các OpenCL  GPU, chẳng hạn như các sảm phầm từ AMD, thông qua PlaidML Keras backend.

## 2.5 Keras hỗ trợ huấn luyện trên nhiều GPU phân tán

+ Keras tích hợp cho dữ liệu song song trên nhiều GPU
+ Horovod, từ Uber, có hỗ trợ tốt nhất cho các mô hình xây dựng trên nền Keras
+ Các mô hình trên nền Keras có thể được chuyển thành các Tensorflow Estimators được huấn luyện trên các cụm GPU trên Google Cloud
+ Keras có thể chạy được trên Spark thông qua Dist-Keras (từ CERN) và Elephas

## 2.6 Cách cài đặt và xây dựng mô hình Keras

### 2.6.1 Hướng dẫn cài đặt Keras trên nền Tensorflow cho Ubuntu (16.04)

Bước 1: Cài đặt Tensorflow
+ pip install --upgrade tensorflow

Bước 2: Kiểm tra Tensorflow đã được cài đặt đúng hay chưa

---
```python
$ python
>>> import tensorflow
>>>
```
---
Bước 3: Cài đặt Keras
+ pip install numpy scipy
+ pip install scikit-learn
+ pip install pillow
+ pip install h5py
+ pip install keras

Bước 4: Kiểm ra tệp. keras.json đã được thiết lập đúng hay chưa.

---
```python
cat ~/.keras/keras.json
{
    "epsilon": 1e-07, 
    "floatx": "float32", 
    "image_data_format": "channels_last", 
    "backend": "tensorflow"
}
```
---
Cụ thể, bạn cần đảm bảo rằng image\_data\_format được đặt thành "channels\_last" (chiều của bức ảnh tuân theo chuẩn của TensorFlow thay vì "channels\_first" cho Theano). Bạn cũng cần đảm bảo rằng keras được chạy trên nền Tensorflow (thay vì Theano).

### 2.6.2 Hướng dẫn cài đặt Keras trên nền Tensorflow cho Windows (7-64bit)

Bước 1: Cài đặt Tensorflow với pip

Nếu một trong số các phiên bản Python sau chưa được cài đặt trên máy của bạn, bạn nên cài đặt nó ngay:
+ Python 3.5.x 64-bit
+ Python 3.6.x 64-bit

TensorFlow hỗ trợ Python 3.5.x and 3.6.x cho Windows. Lưu ý rằng, Python 3 bao gồm gói pip3 hỗ trợ bạn cài đặt Tensorflow một cách dễ dàng. Để cài đặt Tensorflow, khởi động cửa sổ cmd tại thư mục có chứ tệp pip3. Sau đó sử dụng câu lện sau để cài đặt Tensorflow:
+ Phiên bản CPU: pip3 install --upgrade tensorflow
+ Phiên bản GPU: pip3 install --upgrade tensorflow-gpu
Bước 2: Kiểm tra Tensorflow đã được cài đặt đúng hay chưa

---
```python
$ python
>>> import tensorflow
>>>
```
---
Bước 3: Cài đặt Keras
+ pip3 install numpy scipy
+ pip3 install scikit-learn
+ pip3 install pillow
+ pip3 install h5py
+ pip3 install keras
Bước 4: Kiểm ra keras đã được cài hay chưa.

---
```python
>>> from keras import backend
Using TensorFlow backend.
>>>
```
---

## 2.7 Pilot code cho 1 dự án Keras


```python
import tensorflow
from keras import backend
from keras.models import Sequential
from keras.layers.core import Dense, Activation
epoch = 20
batch_size = 128

# 1. data preparation 
X_train = X_train.reshape(60000,784)
y_train = np_utils.to_categorical(y_train,10)

# 2. build the network 
model = Sequential()
model.add(Activation('tanh'))

model.summary()

# 3. specify loss and metrics 
model.compile(loss='categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

# 4. train the model 
model.fit(X_train,y_train,batch_size=128,verbose = 1,epochs = 100,validation_split = 0.8)

# model_yaml = model.to_yaml()
# yaml_file = open('model.yaml', 'w')
# yaml_file.write(model_yaml)
# model.save_weights("model.h5")

yaml_file = open('model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
loaded_model.load_weights('model.h5')
print('a')
```


```python
print 'hello'
```


```python

```


```python

```
