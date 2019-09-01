from numpy import *
import matplotlib.pyplot as plt
from time import sleep
import json
import urllib.request as req
import socket

def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def standRegress(xArr, yArr):      # 标准线性回归函数
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:   # 计算X.T和X乘积的行列式值是否为0，为0则不可逆
        return
    ws = xTx.I * (xMat.T * yMat)   #X.I 表示X的逆
    return ws

def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))    # 对角矩阵
    for j in range(m):        # 按行计算
        diffMat = testPoint - xMat[j,:]
        weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)   # 缺点：对于每一个数据点，都要运行整个数据集
    return yHat

def rssError(yArr, yHatArr):   # 计算实际值与估计值的均方误差
    return ((yArr - yHatArr)**2).sum()


# 岭回归, 通过加lambda的偏置，解决特征比数据点多的问题

def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:
        return
    ws =denom.I * (xMat.T * yMat)
    return ws

def ridgeTest(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean   # 数据标准化
    xMeans = mean(xMat, 0)
    xVar = var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30   # 计算30个不同的λ下的权值
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, exp(i-10))
        wMat[i,:] = ws.T
    return wMat

def regularize(xMat):
    inMat = xMat.copy()
    inMeans = mean(inMat,0)   # 按列求均值
    inVar = var(inMat,0)  # 按列求方差
    inMat = (inMat - inMeans)/inVar
    return inMat

# 前向逐步回归 ， 每一步尽可能减少误差
def stageWise(xArr, yArr, eps=0.01, numIt=100):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m, n = shape(xMat)
    returnMat = zeros((numIt, n))
    ws = zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = inf
        for j in range(n):    # 对每个特征
            for sign in [-1, 1]:  # 增大或减小
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:] = ws.T
    return returnMat


def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    sleep(10)  # 休眠10秒钟，防止短时间内过多的API调用
    # socket.setdefaulttimeout(10)
    myAPIstr = 'AIzaSyBkcAjrM0kwX3i41DTaJWxIGbRfC5LYnvs'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (
    myAPIstr, setNum)
    pg = req.urlopen(searchURL)
    retDict = json.loads(pg.read())  # 用json.load()方法打开和解析url页面内容，得到一部字典，找出价格和其他信息
    pg.close()
    for i in range(len(retDict['items'])):  # 遍历所有条目
        try:
            currItem = retDict['items'][i]  # 获得当前条目
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else:
                newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if sellingPrice > origPrc * 0.5:
                    print("%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except:
            print('problem with item %d' % i)

def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)

'''
交叉验证测试岭回归（用缩减法确定最佳回归系数）：可以观察到缩减程度，同时可以帮助选取主要的特征
@param xArr：数据集，list对象
@param yArr：目标值，list对象
@param numVal：交叉验证的次数
'''

def crossValidation(xArr, yArr, numVal=10):
    m = len(yArr)
    indexList = range(m)
    errorMat = zeros((numVal, 30))
    for i in range(numVal):
        trainX = []
        trainY = []
        testX = []
        testY = []
        random.shuffle(indexList)
        for j in range(m):
            if j < m * 0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX, trainY)
        for k in range(30):
            matTestX = mat(testX)
            matTrainX = mat(trainX)
            meanTrain = mean(matTrainX, 0)
            varTrain = var(matTrainX, 0)
            matTestX = (matTestX - matTrainX) / varTrain
            yEst = matTestX * mat(wMat[k,:].T + mean(trainY))   # 测试岭回归的结果
            errorMat[i, k] = rssError(yEst.T.A, array(testY))   # 存储第i次交叉验证第k个λ下的rss值
    meanErrors = mean(errorMat, 0)
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors==minMean)]  # nonzero(condition)返回满足条件不为0的下标，找到最小权值向量
    xMat = mat(xArr)
    yMat = mat(yArr).T
    meanX = mean(xMat, 0)
    varX = var(xMat, 0)
    unReg = bestWeights / varX  # 因为岭回归使用了归一化，使用要数据还原，即非归一化下的权值
    print('best model:',unReg)
    print(-1 * sum(multiply(meanX, unReg)) + mean(yMat))  # 求解常数项 x0 = -wx + y
def main():
    xArr, yArr = loadDataSet('ex0.txt')
    # 标准线性回归绘图
    # ws = standRegress(xArr, yArr)  # y = ws[0] + ws[1]*X
    # xMat = mat(xArr)
    # yMat = mat(yArr)
    # yHat = xMat * ws
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])
    # xCopy = xMat.copy()
    # xCopy.sort(0)
    # yHat = xCopy * ws
    # ax.plot(xCopy[:,1], yHat)
    # plt.show()

    # 局部加权线性回归绘图
    # yHat = lwlrTest(xArr, xArr, yArr, 0.01)
    # xMat = mat(xArr)
    # print(rssError(yArr, yHat))
    # srtInd = xMat[:,1].argsort(0)
    # xSort = xMat[srtInd][:, 0, :]
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(xSort[:, 1], yHat[srtInd])
    # ax.scatter(xMat[:,1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')
    # plt.show()

    # 岭回归
    # abX, abY = loadDataSet('abalone.txt')
    # ridgeWeight = ridgeTest(abX, abY)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(ridgeWeight)
    # plt.show()

    # 由于乐高api已关闭，无法测试此例
    # lgX = []
    # lgY = []
    # setDataCollect(lgX, lgY)
    # print(lgX, lgY)


if __name__ == '__main__':
    main()