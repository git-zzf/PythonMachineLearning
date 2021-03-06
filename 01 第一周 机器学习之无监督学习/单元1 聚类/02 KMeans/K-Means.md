[TOC]

# K-means方法及应用

k-means算法以k为参数，把n个对象分成k个簇，使簇内具有较高的相似度，而簇间的相似度较低



## 思路

+ 随机选择k个点作为初始的聚类中心
+ 对于剩下的点，根据其与聚类中心的距离，将其归入最近的簇
+ 对每个簇，计算所有点的均值作为新的聚类中心
+ 重复2、3步骤直到聚类中心不再发生改变



## 实例：31省份消费水平分类

### 数据介绍

现有1999年全国31个省份城镇居民家庭平均每人全年消费性支出的八个主要变量数据，这八个变量分别是：食品、衣着、家庭设备用品及服务、医疗保健、交通和通讯、娱乐教育文化服务、居住以及杂项商品和服务。利用已有数据，对31个省份进行聚类



### 目的和技术路线

目的：了解1999年各个省份的消费水平在国内的情况

技术路线：`sklearn.cluster.Kmeans`



## 代码

```python
import numpy as np
from sklearn.cluster import KMeans

path = r"E:/Notes/PythonMachineLearning/01 第一周 机器学习之无监督学习/单元1 聚类/01 全国31省消费水平/31省市居民家庭消费水平-city.txt"

def loadData(filePath):
    fr = open(filePath, 'r+') # r+表示读写打开方式
    lines = fr.readlines() # 返回全部行，以列表储存在lines中
    retData = []    # 用来储存数据部分，包含城市名和消费水平
    retCityName = []    # 用来储存城市名称
    for line in lines:
        items = line.strip().split(",") # 数据是用逗号分隔的)
        retCityName.append(items[0]) # 每行数据的第一个元素是城市名
        retData.append([float(items[i]) for i in range(1,len(items))])
    return retData,retCityName


if __name__ == '__main__':
    data,cityName = loadData(path)	
    km = KMeans(n_clusters = 4) 
    label = km.fit_predict(data) 
    expenses = np.sum(km.cluster_centers_, axis=1) 
    CityCluster = [[], [], [], []]
    for i in range(len(cityName)):
        CityCluster[label[i]].append(cityName[i])
    for i in range(len(CityCluster)):
        print("Expenses:%.2f" % expenses[i])
        print(CityCluster[i])
        


    
```



## 代码解释

### 读取文件的函数

```python
def loadData(filePath):
    fr = open(filePath, 'r+') # r+表示读写打开方式
    lines = fr.readlines() # 返回全部行，以列表储存在lines中
    retData = []    # 用来储存数据部分，包含城市名和消费水平
    retCityName = []    # 用来储存城市名称
    for line in lines:
        items = line.strip().split(",") # 数据是用逗号分隔的)
        retCityName.append(items[0]) # 每行数据的第一个元素是城市名
        retData.append([float(items[i]) for i in range(1,len(items))])
    return retData,retCityName
```

#### open()

`open()`函数是Python 用来读取文件的函数，创建一个file对象

`open(file, mode)`

`file`是文件的位置

`mode`是打开方式，只读，读写，二进制方式打开等等

这里的`'r+'`表示打开一个文件用于读写，文件指针会放在文件的开头，即从开头开始读取



#### file对象

对生成的`file`对象有如下操作方法：

+ **`file.read([size])`**：size 未指定则返回整个文件，如果文件大小 >2 倍内存则有问题，f.read()读到文件尾时返回""(空字串)。
+ **`file.readline()`**：返回一行。
+ **`file.readlines([size])`** ：返回包含size行的列表, size 未指定则返回全部行。
+ **`f.close()`** 关闭文件

这里使用的是`.readlines()`方法，逐行读取，并返回为一个列表，存放在lines变量中



#### 分析数据

数据的格式是这样的

```txt
北京,2959.19,730.79,749.41,513.34,467.87,1141.82,478.42,457.64
天津,2459.77,495.47,697.33,302.87,284.19,735.97,570.84,305.08
河北,1495.63,515.90,362.37,285.32,272.95,540.58,364.91,188.63
```

每个元素用逗号分隔，一个城市的数据占一行，最开始的元素是城市名

提取信息：

```python
 for line in lines:
        items = line.strip().split(",") # 数据是用逗号分隔的)
        retCityName.append(items[0]) # 每行数据的第一个元素是城市名
        retData.append([float(items[i]) for i in range(1,len(items))])
```

逐行操作，所以使用一个循环，对全部行的列表lines中的每一行进行操作

通过`.strip()`去除空格，接着通过`.split(",")`把用逗号分隔的元素逐一提取出来

把第一个元素，也就是城市名提取出来，存放在`retCityName`变量中

把之后的元素存放在`retData`变量中，这里需要把元素变成浮点数储存

这里循环的使用方式是列表解析式，把for循环放在列表中用来生成一个列表，添加到`retData`变量中，经过循环，`retData`变量会变成一个二维列表，类似于[[], [], [], []]

`retData`变量包含所有消费水平的数据信息，`retCityName`变量包含所有城市名



#### 返回结果

```python
return retData,retCityName
```

返回`retData`变量和`retCityName`变量



### 聚类函数部分

```python
if __name__ == '__main__':
    data,cityName = loadData(path)	# 读取数据
    km = KMeans(n_clusters = 4) 	# 调用KMeans函数
    label = km.fit_predict(data) 	# 分类算法，返回值是每个城市对应的类别名称，从0到3编号
    expenses = np.sum(km.cluster_centers_, axis=1) # 计算8种消费水平的总和，用来展示结果
    CityCluster = [[], [], [], []]	# 使用二维列表接收分类后的城市名，共有4类
    for i in range(len(cityName)):
        CityCluster[label[i]].append(cityName[i])	# 把每个城市的名称添加到对应的类别中
    for i in range(len(CityCluster)):	# 按类别打印消费水平和类别中的城市名
        print("Expenses:%.2f" % expenses[i])
        print(CityCluster[i])
```



#### KMeans()函数的设置

```python
sklearn.cluster.KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=None, algorithm='auto')
```

重要参数：

+ `n_clusters`：簇数，标签数，分成几类，默认为8个类别
+ `init`：分类中心初始位置，默认为`k-means++`，自动设置中心位置
+ `n_init`：初始化`KMeans`算法需要尝试多少个中心位置，默认10次
+ `max_iter`：最大迭代次数，默认300次
+ `tol`：阈值，默认1e-4

在这里只需要更改`n_clusters`为需要分的类别，这里选4个

把KMeans()函数赋值给变量`km`方便调用



#### 训练KMeans

调用`fit_predict()`函数

```python
fit_predict(X, y=None, sample_weight=None)
```

`X`：需要分类的数据，以矩阵或二维数组的方式给入，行列为样本数*特证数

`y`：不需要考虑

`sample_weight`：对每个样本权重，默认没有权重

我们的数据是	

```txt
北京,2959.19,730.79,749.41,513.34,467.87,1141.82,478.42,457.64
天津,2459.77,495.47,697.33,302.87,284.19,735.97,570.84,305.08
河北,1495.63,515.90,362.37,285.32,272.95,540.58,364.91,188.63
```

共有31个样本，8个特征（8种消费参数），符合参数`X`的要求

`fit_predict()`函数返回的是簇标签，这里使用变量`label`接收



#### 计算消费总和

计算每个类别的中心（即对每个簇的8种消费水平求和）

`km.cluster_centers_`返回的是4个簇的8种特征的中心

```python
array([[2004.785     ,  429.48      ,  347.8925    ,  190.955     ,
         287.66625   ,  581.16125   ,  437.2375    ,  233.09625   ],
       [1525.81533333,  478.672     ,  322.88266667,  232.4       ,
         236.41866667,  457.53133333,  344.81866667,  190.21933333],
       [2549.658     ,  582.118     ,  488.366     ,  268.998     ,
         397.442     ,  618.92      ,  477.946     ,  295.172     ],
       [3242.22333333,  544.92      ,  735.78      ,  405.51333333,
         602.25      , 1016.62      ,  760.52333333,  446.82666667]])
```

共有4个簇，每个簇里包含这个簇的中心，有8个维度，即特征值

我们要做的是对这8个特征值求和

使用`expenses = np.sum(km.cluster_centers_, axis=1) `，这里设置求和的轴为1，表示对每个簇中间的元素求和，存放在`expenses`变量中，表示每个簇的消费平均值，共有4个元素



#### 存放结果

```python
 CityCluster[label[i]].append(cityName[i])
```

`i`表示城市的序号，`CityCluster`变量用来存放结果，分为4个元素，每个元素中用来存放城市名

`label`变量存放的是每个城市对应的类别序号：

```python
array([3, 2, 1, 1, 1, 1, 1, 1, 3, 0, 2, 0, 2, 1, 1, 1, 0, 0, 3, 0, 0, 2,
       0, 1, 0, 2, 1, 1, 1, 1, 1])
```

`CityCluster[label[i]]`用来把每个城市放到对应的类别下

`append(cityName[i])`把城市名称添加到每个元素中



#### 打印结果

```python
 for i in range(len(CityCluster)):
        print("Expenses:%.2f" % expenses[i])
        print(CityCluster[i])
```

`len(CityCluster)`的输出是4，`CityCluster`有4个元素，每个元素内的有数量不同的城市名，`len()`函数返回的是它的元素数量，即4个

按类别打印消费水平，保留两位小数

按类别打印城市名称