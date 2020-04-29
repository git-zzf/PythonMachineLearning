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