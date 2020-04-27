import numpy as np
from sklearn.cluster import KMeans

path = r"E:/Notes/PythonMachineLearning/01 第一周 机器学习之无监督学习/单元1 聚类/31省消费水平/31省市居民家庭消费水平-city.txt"

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
