'''
4，数据集的划分, 以及用knn算法计算结果
'''
import numpy as np
import part1_file2matrix as f2m
import part3_autoNorm as an
import KNN_function as K

# 处理原始数据
datingDataMat, datingLabels = f2m.file2matrix('E:/class/bilibili机器学习笔记（福州大学江灏）/老师发的课件/MachineLearning/KNN算法实战（二）/datingTestSet.txt')

# 数据归一化
normDataSet = an.autoNorm(datingDataMat)

# 设置划分比例 训练集：测试集 = 8:2
m = 0.8

# 数据总量
dataSize = normDataSet.shape[0]

print('数据集总行数：', dataSize)

# 训练集大小
trainSize = int(m*dataSize)
print(trainSize)

# 测试集大小
testSize = int((1-m)*dataSize)
print(testSize)

# knn算法k值
k = 5
# 预测结果错误的个数
error = 0

# 模型预测的错误率
for i in range(testSize):
    # 前面trainSize个数据不用管， 从trainSize+i-1个开始为测试数据
    # 参数1：测试集， 参数2：训练集， 参数3：标签向量， 参数4：选择距离最小的k个点
    result = K.knn(normDataSet[trainSize+i-1, :], normDataSet[0: trainSize, :], datingLabels[0: trainSize], k)

    # 预测的结果不等于真实结果
    if result != datingLabels[trainSize+i-1]:
        error = error + 1
# 打印错误率
print('rate of the error rusult:', error/testSize)