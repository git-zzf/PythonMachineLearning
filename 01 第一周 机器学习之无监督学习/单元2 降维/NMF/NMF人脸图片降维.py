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
