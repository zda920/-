from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(filename, delim='\t'):
    # 使用两个list构建矩阵
    fr = open(filename)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    dataArr = [list(map(float, line)) for line in stringArr]
    return mat(dataArr)

def pca(dataMat, topNfeat=999999):
    meanVals = mean(dataMat, axis=0)  # 按列求和得到每个特征的平均值
    meanRemoved = dataMat - meanVals
    covMat = cov(meanRemoved, rowvar=0)   # 计算协方差矩阵，rowvar=0，代表一行代表一个样本，计算出的类型是array
    eigVals, eigVects = linalg.eig(mat(covMat))  # 使用linalg.eig()函数，求解协方差矩阵的特征值和特征向量
    eigValInd = argsort(eigVals)  # 从小到大排序，返回排序索引
    eigValInd = eigValInd[: -(topNfeat+1) : -1]  # 得到前topNfeat大的特征向量索引，(n,topNfeat)
    redEigVects = eigVects[:, eigValInd]  # 得到前topNfeat大的特征向量
    lowDDataMat = meanRemoved * redEigVects  # 将数据转化到新的空间，得到降维后的矩阵，(m,topNfeat)
    reconMat = (lowDDataMat * redEigVects.T) + meanVals   # 在新空间下，重构原始数据矩阵，(m,n)
    return lowDDataMat, reconMat

def replaceNanWithMean():
    # 将数据集中的缺失值用平均值代替
    dataMat = loadDataSet('secom.data', ' ')
    numFeat = shape(dataMat)[1]
    for i in range(numFeat):
        meanVal = mean(dataMat[nonzero(~isnan(dataMat[:, i].A))[0], i])
        dataMat[nonzero(isnan(dataMat[:,i].A))[0], i] = meanVal
    return dataMat


def main():
    # dataMat = loadDataSet('testSet.txt')
    dataMat = replaceNanWithMean()
    lowMat, reconMat = pca(dataMat,6)
    print(shape(lowMat))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:,1].flatten().A[0], marker='^', s=90)
    ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='o', s=50, c='red')
    plt.show()

if __name__ == '__main__':
    main()