'''
第一部分
处理原始数据，原始数据转换为矩阵格式，同时提取标签向量
file2matrix函数说明：
打开并解析文件，对数据进行分类：1代表不喜欢，2代表魅力一般，3代表极具魅力
输入参数：filename - 文件名
返回：
returnMat - 特征矩阵
classLabelVector - 分类标签向量
'''

import numpy as np

def file2matrix(filename):

    fr = open(filename)  # 打开文件

    numberOfLines = len(fr.readlines())  # readlines会读出所有行

    returnMat = np.zeros((numberOfLines, 3))  # 创建零矩阵，文件行数 x 3列

    classLabelVector = [] # 定义标签向量

    # 接下来一行一行读数据
    fr = open(filename)  # 同样，先打开文件

    index = 0   # 定位到第一行

    # for循环一行一行读取数据
    for line in fr.readlines():

        line = line.strip()  # 通过strip()函数把每行的数据清理掉空格，数据进行对齐

        listFormLine = line.split('\t')  # 把数据切分出来（根据指定的分隔符对数据进行切分），切分后会保存一个字符串列表

        returnMat[index,:] = listFormLine[0:3] # 把数据的前三列切分出来保存到returnMat（1000x3的矩阵）中
        #如果数据的最后一列为didntLike, 标签向量中对应的元素设置为1
        if listFormLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFormLine[-1] == 'smallDoses':  # 用2来代表魅力一般
            classLabelVector.append(2)
        elif listFormLine[-1] == 'largeDoses':  # 用3代表极具魅力
            classLabelVector.append(3)
        index += 1
    # 返回清洗好的数据，即转为矩阵形式  和   标签向量
    return returnMat, classLabelVector

# 对file2matrix函数进行测试

datingDataMat, datingLabels = file2matrix('E:/class/bilibili机器学习笔记（福州大学江灏）/老师发的课件/MachineLearning/KNN算法实战（二）/datingTestSet.txt')

#打印结果
print(datingDataMat)
print(datingLabels)