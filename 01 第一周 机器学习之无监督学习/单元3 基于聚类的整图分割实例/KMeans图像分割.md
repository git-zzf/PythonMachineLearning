[TOC]

# 基于聚类的图像分割

## 定义

利用图像的灰度、颜色、纹理、形状等特征，把图像分成若干个互不重叠的区域，并使这些特征在同一区域内呈现相似性，在不同的区域之间存在明显差异性。然后就可以将分割的图像中具有独特性质的区域提取出来用于不同的研究



## 应用

机车检测领域：轮毂裂纹图像的分割，及时发现裂纹，保证行车安全

生物医学工程：对肝脏CT图像进行分割，为临床治疗和病理学研究提供帮助



## 常用方法

### 阈值分割

对图像灰度值进行度量，设置不同类别的阈值，达到分割的目的



### 边缘分割

对图像边缘进行检测，即检测图像中灰度值发生跳变的地方，则为一片
区域的边缘



### 直方图法

对图像的颜色建立直方图，而直方图的波峰波谷能够表示一块区域的颜
色值的范围，来达到分割的目的



### 特定理论

基于聚类分析、小波变换等理论完成图像分割



## 实例：KMeans聚类进行图像分割

### 目标

利用KMeans聚类算法对图像像素点颜色进行聚类实现简单的图像分割



### 输出

同一聚类中的点使用相同颜色标记，不同聚类颜色不同



### 技术路线

`sklearn.cluster.KMeans`



### 实例数据

任意大小的图片，区分度模型的图片效果较好



## 实例代码

```python
# 导入库
import numpy as np
import PIL.Image as image # 加载PIL包，用于加载创建图片
from sklearn.cluster import KMeans

# 图片路径
path1 = r'E:/Notes/PythonMachineLearning/01 第一周 机器学习之无监督学习/单元3 基于聚类的整图分割实例/图片数据/person.png'
path2 = r'E:/Notes/PythonMachineLearning/01 第一周 机器学习之无监督学习/单元3 基于聚类的整图分割实例/图片数据/bull.jpg'
pathAfter1 = r'E:/Notes/PythonMachineLearning/01 第一周 机器学习之无监督学习/单元3 基于聚类的整图分割实例/分割后的图片/person.jpg'
pathAfter2 = r'E:/Notes/PythonMachineLearning/01 第一周 机器学习之无监督学习/单元3 基于聚类的整图分割实例/分割后的图片/bull.jpg'

# 加载训练数据
def loadData(filePath):
    f = open(filePath, 'rb')   # 以二进制形式打开文件
    data = []
    img = image.open(f)        # 以列表形式返回图片像素值
    m,n = img.size             # 获得图片的大小
    if filePath[-3:]=='png':       # png文件读取像素点有4个返回值，第四个返回值是像素最大取值范围
        for i in range(m):         # 将像素点归一化，储存在data中
            for j in range(n):
                x, y, z, lum = img.getpixel((i,j))  
                data.append([x/256.0, y/256.0, z/256.0])
    else:
        for i in range(m):         # 将像素点归一化，储存在data中
            for j in range(n):
                x, y, z = img.getpixel((i,j))  
                data.append([x/256.0, y/256.0, z/256.0])
    f.close()
    return np.mat(data), m, n          # 以矩阵返回data，以及图片大小
 
imgData, row, col = loadData(path2)     # 加载数据

# 加载KMeans聚类算法
km = KMeans(n_clusters=3) # 聚类的中心设置为3

# 进行聚类运算
# 获取每个像素所属的类别，并且以图片的大小排列
label = km.fit_predict(imgData)
label = label.reshape([row,col]) 

# 创建一张新的灰度图片保存聚类的结果
pic_new = image.new('L', (row, col))

# 根据所属类别向图片中添加灰度值
for i in range(row):
    for j in range(col):
        pic_new.putpixel((i, j), int(256/(label[i][j]+1)))

# 以JPEG格式保存图像
pic_new.save(pathAfter2, 'JPEG')
```

