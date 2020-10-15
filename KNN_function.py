'''
函数说明：KNN算法分类器
输入参数：
    inX  - 未知样本（测试集）
    dataSet - 训练样本的特征矩阵（训练集）
    labels - 训练集分类标签向量
    k - 选择距离最小的k个点
返回：
maxType - 分类结果

'''

import numpy as np

def knn(inX, dataSet, labels, k):
    # 1.计算距离
    dist = ((dataSet-inX)**2).sum(1)**0.5  # 欧拉公式 -- 欧式距离
    # 2.排序
    sortedDist = dist.argsort()
    # 3.计数
    classCount = {}

    for i in range(k):
        voteLabel = labels[sortedDist[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    # 求最多类型
    maxType = 0

    maxCount = 1

    for key, value in classCount.items():
        if value > maxCount:
            maxType = key
            maxCount = value

    return maxType