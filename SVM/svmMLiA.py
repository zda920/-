from numpy import *
def loadData(filename):
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

def selectJrand(i, m):
    j = i;
    while(j == i):
        j = int(random.uniform(0, m))
    return j

def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

'''
简化版SMO函数
'''

def smoSimple(dataIn, classLabels, C, toler, maxIter):

    """

    :param dataIn:
    :param classLabels:
    :param C:常数C
    :param toler:容错率
    :param maxIter:退出前最大的循环次数
    :return:
    """

    dataMatrix = mat(dataIn)
    labelMat = mat(classLabels).transpose()
    b = 0
    m, n = shape(dataMatrix)
    alphas = mat(zeros((m, 1)))   # 创建一个包含m个alpha值的向量，全部初始化为0
    iter = 0  # 存储没有任何alpha改变的情况下遍历数据集的次数
    while(iter < maxIter):      # 外循环，迭代次数小于最大迭代次数
        alphaPairsChanged = 0   # 记录alpha是否已经进行优化
        for i in range(m):    # 内循环，对数据集中的每个数据向量
            # 计算预测类别 fXi
            # multiply 实现对应位置元素相乘
            fXi = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            # 计算误差
            Ei = fXi - float(labelMat[i])
            # 如果误差很大，则该数据实例所对应的alpha值要进行优化，包括正间隔和负间隔
            # 这里本来是 C>=α>=0，不取等号是因为等号代表数据实例位于边界上，不值得优化
            if((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                # 随机选择第二个alpha值 alphas[j]
                j = selectJrand(i, m)
                fXj = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                # 用copy方法拷贝旧值，防止Python传递列表采用的是传引用方式，进行后面的比较
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if(labelMat[i] != labelMat[j]):  # 控制 alphas[j] 调整到0到C之间
                    L = max(0, alphas[j] - alphas[i])  # 下届不能小于0
                    H = min(C, C + alphas[j] - alphas[i])   # 上界不能大于C
                else:
                    L = max(0, alphas[j] + alphas[i] -C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    # 如果上下界相等，不做任何改变，continue进行下一次循环
                    print('H == L')
                    continue
                # eta是alphas[j]的最优修改量，因为如果eta为0，计算新的alphas[j]比较麻烦，所以对SMO做了简化，不做处理，continue进行下一次循环
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T - dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    print('eta >= 0')
                    continue
                # 对alphas[j]的上下界进行调整
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                # 检查alpha[j]是否有轻微改变，如果是，continue进行下一次循环
                if(abs(alphas[j] - alphaJold) < 0.00001):
                    print('j not moving enough')
                    continue
                # 更新alphas[i]值，与alphas[j]值大小一样，但方向相反，即一个增加，另外一个减少
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                # 设置常数项b
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j,:].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]) :
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1  # 标记这一对alpha值进行了改变
                print('iter：', iter, i, 'pairs changed', alphaPairsChanged)
        if(alphaPairsChanged == 0):
            iter += 1  # 如果这一对alpha值没有改变，则继续迭代。这里是所有alpha不发生修改，程序才会停止退出while循环
        else:
            iter = 0  # 如果这一对alpha值改变了，iter设置为0重新进行迭代
        print('iteration number:', iter)
    return b, alphas

'''
完整版Platt SMO的支持函数
'''

class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))          # 第一列为有效的标志位

def calcEK(oS, k):          # 计算E值
    fXK = float(multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T)) + oS.b
    Ek = fXK - float(oS.labelMat[k])
    return Ek

def selectJ(i, oS, Ei):    # 选择第二个alpha
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]    # 设为有效
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]   # 返回非0E值对应的alpha值
    if(len(validEcacheList)) > 1:
        for k in validEcacheList:   # 选择具有最大步长的j
            if k == i:
                continue
            Ek = calcEK(oS, k)
            deltaE = abs(Ei - Ek)
            if(deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEK(oS, j)
        return j, Ej

def updateEk(oS, k):    # 计算误差值存入缓存
    Ek = calcEK(oS, k)
    oS.eCache[k] = [1, Ek]

def innerL(i, oS):
    Ei = calcEK(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei) #this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H:
            print("L==H")
            return 0
        eta = 2.0 * oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - oS.X[j,:]*oS.X[j,:].T   # changed for kernel
        if eta >= 0:
            print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])       # update i by the same amount as j
        updateEk(oS, i) #added this for the Ecache              # the update is in the oppostie direction
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[i, :].T - oS.labelMat[j] * (
                    oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - oS.labelMat[j] * (
                    oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2)/2.0
        return 1
    else:
        return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    # 集合的一次遍历为一个迭代
    while(iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
            print('fullSet, iter:', iter, 'i:', i, ',pairs changed', alphaPairsChanged)
            iter += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:   # 遍历非边界的alpha值
                alphaPairsChanged += innerL(i, oS)
                print('non-bound,iter:', iter,' i', i, ' pairs changed', alphaPairsChanged)
            iter += 1
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
        print('iter number:', iter)
    return oS.b, oS.alphas

def calcWs(alphas, dataArr, classLabels):  # 对于alphas值中，非0的才是支持向量
    X = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i,:].T)
    return w

'''
使用核函数的svm
'''

def kernelTrans(X, A, kTup):
    m, n = shape(X)
    K = mat(zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow * deltaRow.T
        K = exp(K / (-1 * kTup[1]**2))
    else:
        raise NameError('kernel is not recongnized')
    return K

class optStructK():
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i,:], kTup)    # 填充K矩阵  按行进行转化


def calcEk(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek


def innerLK(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)  # this has been changed from selectJrand 选择最大步长的j
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L==H")
            return 0
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]  # changed for kernel
        if eta >= 0:
            print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)  # added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])  # update i by the same amount as j
        updateEk(oS, i)  # added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smoPK(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):  # full Platt SMO
    oS = optStructK(mat(dataMatIn), mat(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True;
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:  # go over all
            for i in range(oS.m):
                alphaPairsChanged += innerLK(i, oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:  # go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerLK(i, oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False  # toggle entire set loop
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas

def testRbf(k1 = 1.3):
    dataArr, labelArr = loadData('testSetRBF.txt')
    b, alphas = smoPK(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print(shape(sVs)[0], '个支持向量')
    m, n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i,:],('rbf',k1))
        pred = kernelEval.T * multiply(labelSV, alphas[svInd] + b)
        if sign(pred) != sign(labelArr[i]):
            errorCount += 1
    print('error rate:',float(errorCount) / m)
    dataArr, labelArr = loadData('testSetRBF2.txt')
    errorCount = 0
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i,:],('rbf', k1))
        pred = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(pred) != sign(labelArr[i]):
            errorCount += 1
    print('error rate:',float(errorCount) / m)

'''
基于SVM的数字识别（只用于二元分类1和9）
'''

def img2vector(filename):
    returnVect = zeros((1,1024))  # 一副图像是1个1行1024列的向量
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)      # 加载训练集或测试集名称
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     # 去除 .txt 后缀
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9: hwLabels.append(-1)  # 数字9这一类用-1表示
        else: hwLabels.append(1) # 数字1这一类用1表示
        trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))  # 保存图像
    return trainingMat, hwLabels    # 返回图像矩阵、标签

def testDigits(kTup=('rbf', 10)):  # 可以在这里修改内核函数和σ的值
    dataArr,labelArr = loadImages('trainingDigits')
    b,alphas = smoPK(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat = mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A>0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd];
    print("there are %d Support Vectors" % shape(sVs)[0])
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict = kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m))
    dataArr,labelArr = loadImages('testDigits')
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print("the test error rate is: %f" % (float(errorCount)/m))


def main():
    #dataArr, labelArr = loadData('testSet.txt')
    #b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    #b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)
    #ws = calcWs(alphas, dataArr, labelArr)
    #print(ws)
    #print(b, alphas)
    #dataMat = mat(dataArr)
    #testPoint = dataMat[0] * mat(ws) + b
    #print(testPoint, labelArr[0])  # 大于0表示1类，反之-1类
    #testRbf(k1=1.3)
    testDigits()

if __name__ == '__main__':
    main()