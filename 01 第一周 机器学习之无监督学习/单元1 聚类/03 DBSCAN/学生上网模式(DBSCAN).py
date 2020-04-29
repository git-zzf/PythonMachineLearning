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