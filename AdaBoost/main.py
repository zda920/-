from numpy import *

def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):

    """
    通过阈值将数组数据进行分类
    :param dataMatrix:数据集
    :param dimen:第i个特征
    :param threshVal:阈值
    :param threshIneq:不等号(＜、＞)
    :return:
    """

    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}   # 存储给定向量D时所得到的的最佳单层决策树的相关信息
    bestClasEst = mat(zeros((m, 1)))
    minError = inf
    for i in range(n):        # 所有特征上遍历
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)   # 从最小值到最大值滑动
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr   # 错误向量与权重向量D相应元素相乘并求和
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst

def adaBoostTrainsDS(dataArr, classLabels, numIt = 40):

    # numIt 迭代次数

    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m, 1)) / m)   # 数据点的权重，所有元素之和为1.0
    aggClassEst = mat(zeros((m, 1)))  # 数据点的类别估计累计值
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)   #返回利用D得到的最小错误率的单层决策树
        #print('D:', D.T)
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))  # 本次单层决策树输出结果的权重。并防止除零溢出
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        #print('classEst:', classEst.T)
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)   # 为下一次迭代计算权重向量D
        D = multiply(D, exp(expon))
        D = D / D.sum()
        aggClassEst += alpha * classEst
        #print('aggClassEst:', aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum() / m
        #print('error:', errorRate)
        if errorRate == 0.0:
            break
    #return weakClassArr
    return weakClassArr, aggClassEst

def adaClassify(dataClass, classifyArr):
    dataMat = mat(dataClass)
    m = shape(dataMat)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifyArr)):
        classEst = stumpClassify(dataMat, classifyArr[i]['dim'], classifyArr[i]['thresh'], classifyArr[i]['ineq'])
        aggClassEst += classifyArr[i]['alpha'] * classEst  # 输出类别估计值乘上该层决策树的alpha权重
        #print(aggClassEst)
    return sign(aggClassEst)

def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def plotROC(predStrengths, classLabels):
    # predStrength  numpy数组或行向量组成的矩阵，通过sign函数得到
    import matplotlib.pyplot as plt
    cur = (1.0, 1.0) # 保留绘制光标的位置
    ySum = 0.0   #　计算ＡＵＣ的值，累加每点的高度
    numPosClas = sum(array(classLabels) == 1.0)   # 计算正例的数目
    yStep = 1 / float(numPosClas)   # y轴步长
    xStep = 1 / float(len(classLabels) - numPosClas)
    sortedIndicies = predStrengths.argsort()   # 得到矩阵中每个元素的排序索引（从小到大）
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    # loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]: # 将矩阵转化为列表
        if classLabels[index] == 1.0:  # 每label为1.0的数据，沿y轴下降一个步长，降低真阳率
            delX = 0
            delY = yStep
        else:  # 降低假阴率
            delX = xStep
            delY = 0
            ySum += cur[1]
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')
        cur = (cur[0] - delX, cur[1] - delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('ROC curve for adaboost')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print('the area under the curve is:', ySum * xStep)

def main():
    # dataMat, classLabels = loadSimpData()
    # classifyArr = adaBoostTrainsDS(dataMat, classLabels, 9)
    # print(classifyArr)
    # dataArr, labelArr = loadDataSet('horseColicTraining2.txt')
    # classifyArr= adaBoostTrainsDS(dataArr, labelArr, 62)
    # testArr, testLabelArr = loadDataSet('horseColicTest2.txt')
    # pred10 = adaClassify(testArr, classifyArr)
    # errArr = mat(ones((67, 1)))
    # errRate = errArr[pred10 != mat(testLabelArr).T].sum() / 67
    # print('error rate:', errRate)
    dataArr, labelArr = loadDataSet('horseColicTraining2.txt')
    classifyArr, aggClassEst = adaBoostTrainsDS(dataArr, labelArr, 62)
    plotROC(aggClassEst.T, labelArr)

if __name__ == '__main__':
    main()