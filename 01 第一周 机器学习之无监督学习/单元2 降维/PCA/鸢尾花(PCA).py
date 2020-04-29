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