[TOC]

# NMF算法

## 定义

非负矩阵分解（Non-negative Matrix Factorization，NMF）是在矩阵中所有元素均为非负数约束条件下的矩阵分解方法



## 基本思想

给定一个非负矩阵V，NMF能够找到一个非负矩阵W和一个非负矩阵H，使得矩阵W和H的乘积近似等于矩阵V中的值：

$V_{n*m}=W_{n*k}*H_{k*m}$

`W矩阵`：举出图像矩阵，相当于从原矩阵V中抽取出来的特征

`H矩阵`：系数矩阵

NMF能够广泛应用于图像分析、文本挖掘和语音处理等领域



## 优化目标

最小化W矩阵H矩阵的乘积和原始矩阵之间的差别，目标函数如下：

$argmin\frac{1}{2}\|X-WH\|^2=\frac{1}{2}\sum(X_{ij}-WH_{ij})^2$



基于KL散度的优化函数，损失函数如下：

$argminJ(W,H)=\sum(X_{ij}ln\frac{X_{ij}}{WH{ij}}-X_{ij}+WH_{ij})$



公式推导

`https://blog.csdn.net/acdreamers/article/details/44663421/`



## sklearn中非负矩阵分解

在sklearn库中，可以使用`sklearn.decomposition.NMF`加载NMF算法，主要参数有：

+ `n_components`：用于指定分解后矩阵的单个维度
+ `init`：W矩阵和H矩阵的初始化方式，默认为`nndsvdar`



## 实例：NMF人脸数据特征提取

### 目标：

使用Olivetti人脸数据共400个，每个数据是64*64大小。由于NMF分解得到的W矩阵相当于从原始矩阵中提取特征，那么就可以使用NMF对400个人脸数据进行特征提取



### 参数设置

设置本次实验的特征数目是6，即k=6

原图片文件总计是4096*400的矩阵，需要对4096的行进行压缩，所以需要对原矩阵进行转置

V：$(4096*400)^T$

W：$400*k$

H：$k*4096$



### 代码部分

```python
# 导入库
import matplotlib.pyplot as plt # 加载matplotlib用于数据的可视化
from sklearn import decomposition # 加载PCA算法包
from sklearn.datasets import fetch_olivetti_faces # 加载Olivetti人脸数据集导入函数
from numpy.random import RandomState #加载RandomState用于创建随机种子

# 实例导入图片数据
n_row, n_col = 2, 3 # 设置图像展示时的排列情况
n_components = n_row * n_col # 计算图片数目
image_shape = (64, 64) # 设置人脸数据图片的大小
dataset = fetch_olivetti_faces(shuffle = True, random_state = RandomState(0))
faces = dataset.data # 加载图片数据，并打乱顺序

# 展示图片
def plot_gallery(title, images, n_col=n_col, n_row=n_row):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row)) # 创建图片，图片大小（单位英寸）
    plt.suptitle(title, size=16) # 设置标题及字号大小
    
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1) # 选择画制的子图
        vmax = max(comp.max(), -comp.min())
        
        plt.imshow(comp.reshape(image_shape), # 将数值归一化，并以灰度图形式显示
                   cmap=plt.cm.gray, 
                   interpolation='nearest', 
                   vmin=-vmax, vmax=vmax)
        plt.xticks(()) # 去除子图的坐标轴标签
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.) # 对子图位置及间隔调整


# NMF算法调用，并与PCA算法比较
estimators = [('Eigenfaces - PCA using randomized SVD', # 名称
               decomposition.PCA(n_components=6, whiten=True)), # PCA算法实例
              ('Non-negative components - NMF', 
               decomposition.NMF(n_components=6, init='nndsvda', 
                                 tol=5e-3))]


# 降维后数据的可视化
for name, estimator in estimators:  # 分别调用PCA和NMF
    estimator.fit(faces)    # 调用PCA或NMF提取特征
    components_ = estimator.components_ # 获取提取的特征
    plot_gallery(name, components_[:n_components]) # 按照固定格式进行排序
plt.show()

```



### 代码说明

#### enumerate函数

`enumerate()`创建一个迭代类型：

```python
>>>seq = ['one', 'two', 'three']
>>>for i, element in enumerate(seq):
    print i, element

>>> 0 one
	1 two
	2 three
```



#### fetch_olivetti_faces函数

```python
fetch_olivetti_faces(data_home=None, shuffle=False, random_state=0, download_if_missing=True, return_X_y=False)
```



#### imshow函数

```python


 plt.imshow(comp.reshape(image_shape), # 将数值归一化，并以灰度图形式显示
                   cmap=plt.cm.gray, 
                   interpolation='nearest', 
                   vmin=-vmax, vmax=vmax)
```

**vmin, vmax** : scalar, optional, default: None

`vmin` and `vmax` are used in conjunction with norm to normalize luminance data. Note if you pass a `norm` instance, your settings for `vmin` and `vmax` will be ignored.