import numpy as np
import matplotlib.pyplot as plt
import part1_file2matrix as f2m
'''
通过图片画出不同特征向量之间的关系
'''
datingDataMat, datingLabels = f2m.file2matrix('E:/class/bilibili机器学习笔记（福州大学江灏）/老师发的课件/MachineLearning/KNN算法实战（二）/datingTestSet.txt')

#打印结果
print(datingDataMat)
print(datingLabels)

# 用所有样本的第一列和第二列进行绘图，颜色种类根据 datingLabels中类型种类来定
# 横轴为里程数，纵轴为游戏时间
# plt.scatter(datingDataMat[:, 0], datingDataMat[:, 1], c=datingLabels)

# 横轴为游戏时间，纵轴为吃冰淇淋
# plt.scatter(datingDataMat[:, 1], datingDataMat[:, 2], c=datingLabels)

# 横轴为里程数，纵轴为吃冰淇淋
plt.scatter(datingDataMat[:, 0], datingDataMat[:, 2], c=datingLabels)

plt.show()