'''
3.数据归一化
归一化的处理方法有很多种，如0-1标准化， Z-score标准化， sigmoid压缩法等等

本次案例采取 0-1标准化
'''
import numpy as np
import part1_file2matrix as f2m

def autoNorm(dataSet):
    # min(0) 中的0代表按列取值
    minVals = dataSet.min(0)

    maxVals = dataSet.max(0)

    ranges = maxVals - minVals
    # 初始化一个零矩阵，零矩阵的行和列由dataSet.shape决定，dataSet.shape返回行和列的数量
    normDataSet = np.zeros(dataSet.shape)

    m = dataSet.shape[0]

    # 0-1标准化的公式
    normDataSet = (dataSet - minVals)/(maxVals - minVals)

    return normDataSet

# 测试归一化函数
datingDataMat, datingLabels = f2m.file2matrix('E:/class/bilibili机器学习笔记（福州大学江灏）/老师发的课件/MachineLearning/KNN算法实战（二）/datingTestSet.txt')

#打印结果
# print(datingDataMat)
# print(datingLabels)

# 数据归一化
normDataSet = autoNorm(datingDataMat)

print(normDataSet)