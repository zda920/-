from  numpy import *
import matplotlib.pyplot as plt

def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])   #X0的值初始化设置为1.0
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

def gradAscent(dataMatIn, classLabels):
    # 使用的是向量
    dataMatrix = mat(dataMatIn)      # 转成numpy  100×3
    labelMat = mat(classLabels).transpose() # 转成numpy，transpose 交换维度(转置）100×1
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500       # 迭代次数
    weights = ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error   # 更新权值
    return weights


def randGrandAscent(dataMatrix, classLabels, numIter=150):
    # 使用的是数组
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))   # 随机选取样本来更新权值系数，较少周期性波动
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])   # 要删除选择的样本
    return weights

def plotBestFit(weights):

    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)   # 矩阵转成数组
    # print(dataArr)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s') # 画散点图
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1) # 步长0.1,返回数组
    y = (-weights[0]-weights[1]*x)/weights[2]  # 0=w0x0+w1x1+w2x2，其中x0=1，y即x2
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('y');
    plt.show()

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5 :
        return 1.0
    else:
        return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt');
    frTest = open('horseColicTest.txt')
    trainingSet = [];
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):  # 21 个特征，最后一个是分类标签，0 和 1 两类
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = randGrandAscent(array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print('error rate:', errorRate)
    return errorRate

def mutliTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print('average errorRate:', errorSum / float(numTests))

def main():
    dataMat, labelMat = loadDataSet()
    wei = gradAscent(dataMat, labelMat)
    # plotBestFit(wei.getA())   # getA()将weights矩阵转换为数组，getA()函数与mat()函数的功能相反
    weight = randGrandAscent(array(dataMat), labelMat, 500)
    # plotBestFit(weight)
    colicTest()

if __name__ == '__main__':
    main()
