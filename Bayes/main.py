from numpy import *

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]  # 每一行代表一篇文档
    classVec = [0, 1, 0, 1, 0, 1]  # 1代表侮辱性文字，0代表正常言论
    return postingList, classVec  # 返回数据集和类别标签

# 创建包含所有不重复单词的列表
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 两个集合的并操作
    return  list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)  # 与vocablist等长,初始值为0
    for word in inputSet:             # 存在则标记为1
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
    #else:
        #print(word, 'not in vocabulary')
    return returnVec

def train(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)   # 文章的数量
    numWords = len(trainMatrix[0])    # vocabList的长度
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = ones(numWords)   # 设为ones 和 2.0 防止其中一个概率值为0，使最后乘积为0
    p1Num = ones(numWords)
    p0Denom = 2.0   #至少两个单词
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]             # 每篇文档每个单词的次数累加
            p1Denom += sum(trainMatrix[i])      # 总的单词数和
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num / p1Denom)     #在分类1中每个单词的概率
    p0Vect = log(p0Num / p0Denom)     # 取对数防止下溢出
    return p0Vect, p1Vect, pAbusive

def classfy(vec2Classfy, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classfy * p1Vec) + log(pClass1)
    p0 = sum(vec2Classfy * p0Vec) + log(1 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def test():
    data, labels = loadDataSet()
    myVocabList = createVocabList(data)
    trainMat = []
    for postin in data:
        trainMat.append(setOfWords2Vec(myVocabList, postin))
    p0V, p1V, pAb = train(array(trainMat), array(labels))
    testEntry1 = ['love', 'my', 'dalmation']
    thisDoc1 = array(setOfWords2Vec(myVocabList, testEntry1))
    print(testEntry1, '分类为：', classfy(thisDoc1, p0V,p1V,pAb))
    testEntry2 = ['stupid', 'garbage']
    thisDoc2 = array(setOfWords2Vec(myVocabList, testEntry2))
    print(testEntry2, '分类为：', classfy(thisDoc2, p0V, p1V, pAb))

def bagOfWordsVecMN(vocabList, iuputSet):   #词袋模型,与setOfWords2Vec函数功能相同
    returnVec = [0] * len(vocabList)
    for word in iuputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def textParse(bigString):
    import re
    listOfTokens = re.split(r'\w*', bigString)        # 正则表达式将字符串划分成每个单词
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainSet = list(range(50))
    testSet = []
    for i in range(10):   # 随机构建训练集 从50个训练文档中随机选择10个文档作为测试集
        randIndex = int(random.uniform(0, len(trainSet)))  # 在[0,50) 内随机生成一个实数，然后将其转化为整数
        testSet.append(trainSet[randIndex])
        del(trainSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = train(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classfy(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('error rate:', float(errorCount) / len(testSet))

def main():
    # data, labels = loadDataSet()
    # myVocabList = createVocabList(data)
    # print(data[0])
    # vec = setOfWords2Vec(myVocabList, data[0])
    # print(myVocabList, vec)
    # trainMat = []
    # for postin in data:
    #     trainMat.append(setOfWords2Vec(myVocabList, postin))
    # print(trainMat)
    # test()
    spamTest()

if __name__ == '__main__':
    main()