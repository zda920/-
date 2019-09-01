from numpy import *
from numpy import linalg as la

# 欧氏距离
# inA, inB必须是列向量，否则会计算错误
def ecludSim(inA,inB):
    return 1.0/(1.0 + la.norm(inA - inB))  # linalg.norm() 用于计算范式 ||A-B||

# 皮尔逊相关系数
def pearsSim(inA,inB):
    if len(inA) < 3 : return 1.0  # 如果向量不存在3个或更多的点，该函数返回1，表示两个向量完全相关
    return 0.5+0.5*corrcoef(inA, inB, rowvar = 0)[0][1]  # [-1,1] 归一化 [0,1]

# 余弦相似度
def cosSim(inA,inB):
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)  # [-1,1] 归一化 [0,1]

def standEst(dataMat, user, simMeas, item):
    # 在给定相识度的计算方法下，用户对物品的估计评分值
    # user：用户编号
    # simMeas：相似度计算方法
    # item：未打分项物品编号
    n = shape(dataMat)[1]
    simTotal = 0.0    # 总相似度
    ratSimTotal = 0.0   # 总的评分（相似度和当前用户评分的乘积）
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0:   # 如果用户没有对某物品评分，则跳过该物品
            continue
        # 找到所有用户对这两种物品都打过分的项的索引
        overLap = nonzero(logical_and(dataMat[:, item].A > 0, dataMat[:,j].A > 0))[0]
        if len(overLap) == 0:
            similarity = 0
        else:
            # 否则，计算两种物品打过分的项的相似度
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal # 将评分值映射到[1,5]之间，返回该未打分项物品的评分

# 使用svd的评分估计，参数解释同standEst
def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    U,Sigma,VT = la.svd(dataMat)   # 进行奇异值分解，降维
    Sig4 = mat(eye(4)*Sigma[:4])   # 前4个奇异值包含总能量的90%，故将矩阵降维成4维
    xformedItems = dataMat.T * U[:,:4] * Sig4.I  # 利用U矩阵将物品转换到低维空间中，即降成4维
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or j==item:   # 加不加 j==item 无所谓，因为item本来就是未评分的物品
            continue
        # 相似度的计算是在低维(4维)空间进行的，注意这里必须是列向量
        similarity = simMeas(xformedItems[item,:].T, xformedItems[j,:].T)
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal

def recommend(dataMat, user, N=3, simMeas=cosSim, estMethond=standEst):
    # 对该用户每一个未评分的物品计算评分，进行排序，产生推荐
    unratedItems = nonzero(dataMat[user,:].A == 0)[1]  # 返回该用户未打过分的物品的索引（列的索引）
    if len(unratedItems) == 0:
        return 'you rated everything'
    itemScores = []
    for item in unratedItems:  # 对每个未评分的物品，使用standEst得到物品的估计得分
        estimatedScore = estMethond(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj:jj[1], reverse=True)[:N]  # 按照物品评分(jj[1])从大到小排序


''' 
基于SVD的图像压缩 
通过由奇异值对矩阵分解，进行降维
'''
def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k]) > thresh:
                print(1,end='') # end默认是'\n'，所以如果不换行，将end设置为''或' '
            else: print(0,end='')
        print()

# @param numSV：奇异值数目，默认为3
def imgCompress(numSV=3, thresh=0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    print("****original matrix******")
    printMat(myMat, thresh)
    U,Sigma,VT = la.svd(myMat)
    SigRecon = mat(zeros((numSV, numSV)))
    for k in range(numSV):  # 构造对角阵，保留前numSV个奇异值
        SigRecon[k,k] = Sigma[k]
    reconMat = U[:,:numSV] * SigRecon * VT[:numSV,:]  # 重构后的矩阵 (m,n)
    print("****reconstructed matrix using %d singular values******" % numSV)
    printMat(reconMat, thresh)

def main():
    dataMat = mat([[4, 4, 0, 2, 2], [4, 0, 0, 3, 3], [4, 0, 0, 1, 1], [1, 1, 1, 2, 0], [2, 2, 2, 0, 0], [1, 1, 1, 0, 0],
                    [5, 5, 5, 0, 0]])
    dataMat2 = mat([[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]])
    # print("cosSim:\n", recommend(dataMat, 2))  # 用户2代表矩阵的第3行
    # print("cosSim:\n", recommend(dataMat2, 1, estMethond=svdEst))

    # imgCompress(2)
if __name__ == '__main__':
    main()