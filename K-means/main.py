from numpy import *
import matplotlib.pyplot as plt
from time import sleep

def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))  # 映射一行元素为float
        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA, vecB):
    # 计算欧氏距离
    return sqrt(sum(power(vecA - vecB, 2)))

def randCent(dataSet, k):
    # 为数据集中每个特征随机创建k个聚簇中心
    n = shape(dataSet)[1]    # 特征数
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = minJ + rangeJ * random.rand(k, 1)  # random.rand(k,1) 是numpy中的函数，随机生成k行1列的(0,1)范围的高斯随机数
    return centroids

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))  # 存储每个点的簇分配结果，第一列记录簇索引值，第二列存储误差（距离的平方）
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:], dataSet[i,:])  # 计算到中心的距离
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            # 如果任何一个簇中心位置发生了改变，那么就更改标志clusterChanged为true
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex, minDist**2
        # print(centroids)
        for cent in range(k):   # 更新质心的位置
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]  # 得到在这个簇中心的所有数据点
            centroids[cent,:] = mean(ptsInClust, axis=0)  # 按列求均值（注意：簇中心不包括随机选择的那个点）
    return centroids, clusterAssment  # 返回簇中心、簇分配结果矩阵

# 为克服k-均值的收敛局部最小值，提出二分k-均值算法
def bikMeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]  # 创建一个初始簇，只有一个簇中心，并转化为列表
    centList = [centroid0]  # 加一层列表包含初始簇列表，以后会越来越多，直到等于k个
    for j in range(m):
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j,:])**2  # 初始误差
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):  # 对每一个簇划分
            ptsIncurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :] # 得到在簇i中的数据点
            centroidMat, splitClustAss = kMeans(ptsIncurrCluster, 2, distMeas)  # 使用kMeans算法将簇一分为二
            sseSplit = sum(splitClustAss[:, 1])  # 计算新划分的簇中数据的误差平方和
            # 计算剩余簇中数据的SSE值，如果是第一次，该值为0，因为所有数据都用来划分新簇，故没有剩余簇的数据
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print(sseSplit, sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat.copy() # 新簇，在原来基础上+1个簇中心，要用copy方法，防止矩阵传引用同时更改bestNewCents的值
                bestClustAss = splitClustAss.copy() # 新的划分结果
                lowestSSE = sseSplit + sseNotSplit
        # 根据上面的划分方法，下面进行实际划分，将要划分的簇中所有点的簇分配结果进行修改
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)   # 为新划分出的簇赋一个新的编号
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print('bestCentToSplit', bestCentToSplit)
        print('len of bestClustAss', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]  # 更新原来的旧簇的值
        centList.append(bestNewCents[1,:].tolist()[0]) # 将新划分出的簇加入centList中
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0],:] = bestClustAss  # 更新clusterAssment
    return mat(centList), clusterAssment  # 返回簇中心、簇分配结果矩阵


'''
对地图上的点进行聚类
书本提供的yahoo api已经失效，可根据已保存的places文本直接读入
'''

# 球面距离计算
def distSLC(vecA, vecB):  # vecA，vecB 是两个经纬度向量
    a = sin(vecA[0, 1] * pi / 180) * sin(vecB[0, 1] * pi / 180)  # pi在numpy中被导入
    b = cos(vecA[0, 1] * pi / 180) * cos(vecB[0, 1] * pi / 180) * \
        cos(pi * (vecB[0, 0] - vecA[0, 0]) / 180)
    return arccos(a + b) * 6371.0  # 返回地球表面两点间的距离

# 簇绘图函数，默认为5个聚类中心，可以按情况修改

def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])  # 每一个位置的经纬度组成一个列表
    datMat = mat(datList)
    myCentroids, clustAssing = bikMeans(datMat, numClust, distMeas=distSLC)  # 使用二分均值算法获得聚簇中心和簇分配结果
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']  # 定义很多个不同的标记形状
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')  # 基于一副图像创建矩阵
    ax0.imshow(imgP)  # 绘制图形矩阵
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)  # 另一套坐标系，绘制数据点和聚类中心
    for i in range(numClust):  # 对于每一个簇
        ptsInCurrCluster = datMat[nonzero(clustAssing[:, 0].A == i)[0], :]  # 选择满足第i个簇的所有数据点
        markerStyle = scatterMarkers[i % len(scatterMarkers)]  # 选择标记形状
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0], marker=markerStyle,
                    s=90)  # 绘制数据点
    ax1.scatter(myCentroids[:, 0].flatten().A[0], myCentroids[:, 1].flatten().A[0], marker='+', s=300)  # 绘制簇中心
    plt.show()  # 显示，最后将绘制结果保存为 result.png


def main():
    dataMat = mat(loadDataSet('testSet2.txt'))
    # centList, myNewAssments = bikMeans(dataMat, 3)
    # print(centList)
    # print(myNewAssments)
    # 由bikMeans返回的k个中心点和分配结果矩阵进行绘图
    # myCentroids, clustAssing = bikMeans(dataMat, 3)
    # fig = plt.figure()
    # rect = [0.1, 0.1, 0.8, 0.8]
    # scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']  # 定义很多个不同的标记形状
    # ax1 = fig.add_axes(rect, label='ax1')  # 另一套坐标系，绘制数据点和聚类中心
    # for i in range(3):  # 对于每一个簇
    #     ptsInCurrCluster = dataMat[nonzero(clustAssing[:, 0].A == i)[0], :]  # 选择满足第i个簇的所有数据点
    #     markerStyle = scatterMarkers[i % len(scatterMarkers)]  # 选择标记形状
    #     ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0], marker=markerStyle,
    #                 s=90)  # 绘制数据点
    # ax1.scatter(myCentroids[:, 0].flatten().A[0], myCentroids[:, 1].flatten().A[0], marker='+', s=300)  # 绘制簇中心
    # plt.show()  # 显示，最后将绘制结果保存为 result.png

    # clusterClubs()

if __name__ == '__main__':
    main()