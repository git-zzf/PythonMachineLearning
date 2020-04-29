[TOC]

# DBSCAN方法及应用

## 定义

DBSCAN算法是一种基于密度的聚类算法

+ 聚类的时候不需要预先指定簇的个数k
+ 最终簇的个数也不确定



## 数据点

DBSCAN将数据点分为3类：

+ 核心点：在半径`Eps`内含有超过`MinPts`数目的点
+ 边界点：在半径`Eps`内点的数量小于`MinPts`，但是落在核心点的邻域内
+ 噪音点：既不是核心点也不是边界点的点



## 算法流程

+ 将所有点标记为核心点、边界点或噪声点
+ 删除噪声点
+ 为距离在Eps之内的所有核心点之间赋予一条边
+ 每组连通的核心点形成一个簇
+ 将每个边界点指派到一个与之关联的核心点的簇中（即哪一个核心点的半径范围之内）



### 举例

有13个样本点，使用DBSCAN进行聚类

|      | P1   | P2   | P3   | P4   | P5   | P6   | P7   | P8   | P9   | P10  | P11  | P12  | P13  |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| X    | 1    | 2    | 2    | 4    | 5    | 6    | 6    | 7    | 9    | 1    | 3    | 5    | 3    |
| Y    | 2    | 1    | 4    | 3    | 8    | 7    | 9    | 9    | 5    | 12   | 12   | 12   | 3    |

+ 取`Eps`=3，`MinPts`=3，依据DBSCAN对所有点进行聚类（曼哈顿距离）

+ 对每个点计算其邻域`Eps`=3内的点的集合
+  集合内点的个数超过`MinPts`=3的点为核心点
+  查看剩余点是否在核心点的邻域内，若在则为边界点，否则就是噪声点
+ 将距离不超过`Eps=3`的点相互连接，构成一个簇，核心点邻域内的点也会被加入到这个簇中



## 学生上网模式实例

### 实例介绍

现有大学校内网的日志数据，290条大学生的校园网使用情况数据，数据包括用户ID，设备的MAC地址，IP地址，开始上网时间，停止上网时间，上网时长，校园网套餐等。利用已有数据，分析学生上网的模式



#### 实验目的

通过DBSCAN聚类，分析学生上网时间和上网时长的模式



#### 技术路线

`sklearn.cluster.DBSCAN`



#### 数据

```txt
2c929293466b97a6014754607e457d68,U201215025,A417314EEA7B,10.12.49.26,2014-07-20 22:44:18.540000000,2014-07-20 23:10:16.540000000,1558,15,本科生动态IP模版,100元每半年,internet
```

数据用逗号分隔开，数据分别为：

```txt
记录编号：2c929293466b97a6014754607e457d68
学生编号：U201215025
MAC地址：A417314EEA7B
IP地址：10.12.49.26
开始上网时间：2014-07-20 22:44:18.540000000
停止上网时间：2014-07-20 23:10:16.540000000
上网时长：1558
```



#### 实验过程：

+ 建立工程，引入相关包

+ 加载数据，预处理数据
+ 上网时长的聚类分析
+ 分析结果
+ 上网时间的聚类分析
+ 分析结果



### 代码

```python
import numpy as np
import sklearn.cluster as skc
from sklearn import metrics
import matplotlib.pyplot as plt
 
path = r"E:/Notes/PythonMachineLearning/01 第一周 机器学习之无监督学习/单元1 聚类/02 学生上网模式/学生月上网时间分布-TestData.txt"
mac2id=dict()	# 生成一个字典
onlinetimes=[]
f=open(path,encoding='utf-8')
for line in f:
    mac=line.split(',')[2]		# 提取MAC地址
    onlinetime=int(line.split(',')[6])	# 提取在线时间，转换成整数型
    starttime=int(line.split(',')[4].split(' ')[1].split(':')[0]) # 提取开始上网时间
    if mac not in mac2id:
        mac2id[mac]=len(onlinetimes)	# 保存MAC地址和MAC地址编号到字典
        onlinetimes.append((starttime,onlinetime))	# 保存开始上网时间和在线时长
    else:
        onlinetimes[mac2id[mac]]=[(starttime,onlinetime)]
real_X=np.array(onlinetimes).reshape((-1,2)) # 把上网时间变量变为两列
X=real_X[:,0:1]	# 选择开始上网时间

db=skc.DBSCAN(eps=0.01,min_samples=20).fit(X) # 开始计算DBSCAN聚类
labels = db.labels_		# 提取标签
raito=len(labels[labels[:] == -1]) / len(labels)	# 计算噪声点占比
print('Noise raito:',format(raito, '.2%'))
 
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)	# 簇数量
print('Estimated number of clusters: %d' % n_clusters_)	
print("Silhouette Coefficient: %0.3f"% metrics.silhouette_score(X, labels))	# 轮廓系数
 
for i in range(n_clusters_):	# 打印每个类别下的开始上网时间
    print('Cluster ',i,':')
    print(list(X[labels == i].flatten()))
     
plt.hist(X,24)
plt.show()
```



### 代码解释

#### 预处理数据

```python
if mac not in mac2id:
        mac2id[mac]=len(onlinetimes)	# 保存MAC地址和MAC地址编号到字典
        onlinetimes.append((starttime,onlinetime))	# 在onlinetimes中添加开始和结束上网时间
    else:
        onlinetimes[mac2id[mac]]=[(starttime,onlinetime)]
```

使用`mac2id`变量储存MAC地址和MAC地址的编号

字典的键是MAC地址，值是从0开始的MAC对应的编号

用`onlinetimes`列表变量中对每个MAC地址编号储存开始上网时间和上网时长



```python
real_X=np.array(onlinetimes).reshape((-1,2)) # 把上网时间变量变为两列
X=real_X[:,0:1]	# 选择开始上网时间
```

用`np.array()`对`onlinetimes`列表建立数组，`.reshape((-1,2))`表示规定列为2，行数不确定

取第一列数据（开始上网时间）进行聚类运算



#### 调用聚类运算

```python
db=skc.DBSCAN(eps=0.01,min_samples=20).fit(X) 
labels = db.labels_
raito=len(labels[labels[:] == -1]) / len(labels)
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
```



DBSCAN重要参数：

+ `eps`： DBSCAN算法参数，即我们的$\epsilon$-邻域的距离阈值，和样本距离超过$\epsilon$的样本点不在$\epsilon$-邻域内。默认值是0.5.一般需要通过在多组值里面选择一个合适的阈值。eps过大，则更多的点会落在核心对象的$\epsilon$-邻域，此时我们的类别数可能会减少， 本来不应该是一类的样本也会被划为一类。反之则类别数可能会增大，本来是一类的样本却被划分开。
+ `min_samples`： DBSCAN算法参数，即样本点要成为核心对象所需要的$\epsilon$-邻域的样本数阈值。默认值是5. 一般需要通过在多组值里面选择一个合适的阈值。通常和eps一起调参。在eps一定的情况下，min_samples过大，则核心对象会过少，此时簇内部分本来是一类的样本可能会被标为噪音点，类别数也会变多。反之min_samples过小的话，则会产生大量的核心对象，可能会导致类别数过少。
+ `metric`：最近邻距离度量参数。可以使用的距离度量较多，一般来说DBSCAN使用默认的欧式距离（即p=2的闵可夫斯基距离）就可以满足我们的需求。可以使用的距离度量参数有：
  + 欧式距离 `euclidean`
  + 曼哈顿距离`manhattan`
  + 切比雪夫距离`chebyshev`
  + 标准化欧式距离`seuclidean`
  + 马氏距离`mahalanobis`



```python
db=skc.DBSCAN(eps=0.01,min_samples=20).fit(X) 
```

`.fit(X)`开始训练模型，结果储存在`db`中



```python
labels = db.labels_
```

查看分类之后的标签



```python
raito=len(labels[labels[:] == -1]) / len(labels)
```

计算噪声所占比例，噪声点在标签中用-1标记



```python
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
```

计算簇的个数，`set(labels)`函数可以把`labels`中所有不同的元素输出出来。把总数减去1（如果包含噪声点-1）或0（不包含噪声点）就可以得到总的簇的数量



#### 打印结果

```python
print("Silhouette Coefficient: %0.3f"% metrics.silhouette_score(X, labels))
```

打印轮廓系数，轮廓系数越接近1说明分类效果越好



```python
for i in range(n_clusters_):
    print('Cluster ',i,':')
    print(list(X[labels == i].flatten()))
```

分别打印每个类别和其中的元素，`X[labels == i]`会在X中把对应类别的元素取出来，`.flatten()`可以把高维数组降到1维数组



```python
plt.hist(X,24)
plt.show()
```

用直方图显示结果，分成24个直方代表每个小时





