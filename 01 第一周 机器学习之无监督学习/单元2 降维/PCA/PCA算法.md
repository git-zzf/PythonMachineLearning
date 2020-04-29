[TOC]

# PCA方法及其应用

## 定义

+ 主成分分析（Principal Component Analysis，PCA）是最常用的一种降维方法，通常用于高维数据集的探索与可视化，还可以用作数据压缩和预处理等

+ PCA可以把具有相关性的高维变量合成为线性无关的低维变量，称为主成分，主成分能够尽可能保留原始数据的信息



## 相关术语回顾

+ 方差
+ 协方差
+ 协方差矩阵
+ 特征向量和特征值



### 方差

各个样本和样本均值的差的平方和的均值，用来度量一组数据的分散程度

$S^2=\frac{\sum_{i=1}^n(x_i-x)^2}{n-1}$



### 协方差

用于度量两个变量之间的线性相关程度，若两个变量的协方差为0，则可认为二者线性无关，协方差矩阵则是由变量的协方差值构成的矩阵（对称阵）

$Cov(X,Y)=\frac{\sum_{i=1}^n(X_i-\overline{X})(Y_i-\overline{Y})}{n-1}$



### 特征向量

矩阵的特征向量是描述数据集结构的非零向量并满足如下公式

$A\overrightarrow{v}=\lambda\overrightarrow{v}$

A是一个方阵，$\overrightarrow{v}$是特征向量，$\lambda$是特征值



## PCA的原理

矩阵的主成分就是其协方差矩阵对应的特征向量，按照对应的特征值大小进行排序，最大的特征值就是第一主成分，其次是第二主成分，以此类推



### 算法过程

**输入：**样本集$D=\left\{x_1,x_2,...,x_m\right\}$

​			低维空间维度 $d^\prime$

**过程：**

1. 对所有样本进行中心化，每个元素减去总的平均值：$x_i\leftarrow x_i-\frac{1}{m}\sum_{i=1}^mx_i$
2. 计算样本的协方差矩阵$XX^T$
3. 对协方差矩阵$XX^T$做特征值分解
4. 取最大的 $d^\prime$个特征值所对应的特征向量$w_1,w_2,...,w_{d^\prime}$

**输出：**投影矩阵 $W=(w_1,w_2,...,w_{d^\prime})$



## 使用sklearn中的PCA函数

### 参数

在sklearn库中，可以使用`sklearn.decomposition.PCA`加载PCA进行降维，主要参数有：

```python
sklearn.decomposition.PCA(n_components=None, copy=True, whiten=False)
```

+ `n_components`：PCA算法中所要保留的主成分个数，也即保留下来的特征个数，或者降维后数据的维度，如果 n_components = 1，将把原始数据降到一维
+ `svd_solver`：设置特征值分解的方法，默认为`auto`
+ `copy`：默认为True，即是否需要将原始训练数据复制
+ `whiten`：默认为False，即是否白化，使得每个特征具有相同的方差



## 鸢尾花数据可视化实例

### 目标

已知鸢尾花数据是4维的共3类样本，使用PCA实现对鸢尾花数据进行降维，实现在二维平面上的可视化



### 代码

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
 
data = load_iris() # 从库中加载数据集
y = data.target	# 从数据集中分别读取target和data
X = data.data
pca = PCA(n_components=2) # 降到2维
reduced_X = pca.fit_transform(X) # 对data部分进行降维操作
print(reduced_X)
red_x, red_y = [], []	
blue_x, blue_y = [], []
green_x, green_y = [], []
 
for i in range(len(reduced_X)):
    if y[i] == 0:		# y表示target部分，即类别，这步是把数据点按类别分开，共有3类
        red_x.append(reduced_X[i][0])	
        red_y.append(reduced_X[i][1])
    elif y[i] == 1:
        blue_x.append(reduced_X[i][0])
        blue_y.append(reduced_X[i][1])
    else:
        green_x.append(reduced_X[i][0])
        green_y.append(reduced_X[i][1])
 
plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()
```



### 注释

返回的`reduced_x`：

```python
[[-2.68412563  0.31939725]
 [-2.71414169 -0.17700123]
 [-2.88899057 -0.14494943]
 [-2.74534286 -0.31829898]
 ...
 [ 1.39018886 -0.28266094]]
```

其中每个元素表示一个二维点的坐标