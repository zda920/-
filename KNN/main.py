import numpy as np
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
from os import listdir

def createData():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    label = ['A', 'A', 'B', 'B']
    return group, label

def classify(inX, dataset, labels, k):
    # calculate dist
    datasetSize = dataset.shape[0]
    diffMat = tile(inX, (datasetSize, 1)) - dataset   # tile在列方向上重复inX datasize次，默认行1次
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)  #多维数组行相加
    distances = sqDistances**0.5
    sortedDistIndicices = distances.argsort()   #从小到大返回下标
    classCount = {}
    # votel k
    for i in range(k):
        votelLabel = labels[sortedDistIndicices[i]]
        classCount[votelLabel] = classCount.get(votelLabel, 0) + 1 # get方法，如果key不存在，可以返回None，或者自己指定的0
    # sort
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 从文件读入数据
def file2matrix(filename):
    fr = open(filename)
    array_lines = fr.readlines()
    number_lines = len(array_lines)  # 文件行数
    returnMat = zeros((number_lines, 3)) # 0填充numpy矩阵
    classLabelVector = []
    index = 0
    for line in array_lines:
        line = line.strip()   # 除去回车
        listFromLine = line.split('\t')  #按tap划分
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

# 数据归一化，防止不同权重的影响
# 将数据转化到0-1
def autoNorm(dataSet):
    min_vals = dataSet.min(0)  # 0表示从列中选最值 shape为1x3
    max_vals = dataSet.max(0)
    ranges = max_vals - min_vals   # 最大值减最小值
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(min_vals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, min_vals

def datingTest():
    hoRatio = 0.20
    dataMat, dataLabels = file2matrix('dataTestSet2.txt')
    normMat, ranges, min_vals = autoNorm(dataMat)
    m = normMat.shape[0]
    numTest = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTest):
        classifyResult = classify(normMat[i,:], normMat[numTest:m,:], dataLabels[numTest:m], 3)
        print('compare:',i , classifyResult, dataLabels[i])
        if(classifyResult != dataLabels[i]):
            errorCount += 1.0
    print('error rate:',errorCount / float(numTest))

def img2vector(filename):
    returnVector = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVector[0,32*i+j] = int(lineStr[j])
    return returnVector

def handwritingTest():
    hwLabels = []
    trainFile = listdir('trainingDigits')
    m = len(trainFile)
    trainMat = zeros((m,1024))
    for i in range(m):
        fileName = trainFile[i]
        classNum = int(fileName.split('.')[0].split('_')[0])
        hwLabels.append(classNum)
        trainMat[i,:] = img2vector('trainingDigits/%s' % fileName)
    testFile = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFile)
    for i in range(mTest):
        fileName = testFile[i]
        classNum = int(fileName.split('.')[0].split('_')[0])
        testMat = img2vector('testDigits/%s' % fileName)
        testResult = classify(testMat, trainMat, hwLabels, 3)
        print('compare:',i , testResult, classNum)
        if(testResult != classNum):
            errorCount += 1.0
    print('error rate:', errorCount / float(mTest))

def main():
    dataset, labels = createData()
    print(classify([0.7,0.8], dataset, labels, 1))
    datingTest()
    handwritingTest()

if __name__ == '__main__':
    main()