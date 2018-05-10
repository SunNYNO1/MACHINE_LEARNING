    
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
import random
import sys
import re
from os import listdir

def creatDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels


'''根据KNN算法进行确定分类:inx是特征值列表，dataset是学习集矩阵，
labels是标记向量列表，k是KNN算法中的前k个值'''
def classify0(inX,dataSet,labels,k):
    # ~ 数组的shape使用：shape输出数组的列数和行数。shape(0)输出行数，shape（1）输出列数
    dataSetSize = dataSet.shape[0]
    # ~ 根据关键字构造新的数组，如tile(A,(3,2)),其中A=（1，2）的结果就是：[[1，2,1，2],[A,A],[A,A]]
    # ~ 即构造3个新数组，每个型数组重复2遍A中的内容
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    # ~ 返回的distances是一个一维列表
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    sortedClassCount = sorted(classCoun5t.iteritems(),key = operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]
'''输入测试点和学习集，判断分类'''
# ~ print classify0([0.01,0],array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]]),['A','A','B','B'],3)


'''从文本中解析数据为特征列表和标记列表，即格式化数据'''
def file2matrix(filename):
    f = open(filename)
    arrayLines = f.readlines()
    numLines = len(arrayLines)
    # ~ 生成numlines行，3列的零矩阵框架
    mat = zeros((numLines,3))
    classLabelVector = []
    index = 0
    for line in arrayLines:
        line = line.strip()
        listFromLine = line.split(',')
        i = 0
        for item in listFromLine:
            if item != '':
                item = item.strip()
                listFromLine[i] = int(item)
                i += 1
        mat[index,:] = listFromLine[0:3]
        classLabelVector.append(listFromLine[-1])
        index += 1
    return mat, classLabelVector
'''自动生成1000个数据，并导入文件夹'''
# ~ filename = raw_input(unicode('请输入文件路径：','utf-8').encode('gbk','ignore'))
# ~ f = open(filename,'a')
# ~ sys.stdout = f
# ~ for num in range(1,1001):
    # ~ time_percent = random.choice(range(0,100))
    # ~ distance_fly = random.choice(range(0,150000))
    # ~ classify = random.choice([1,2,3,4])
    # ~ print num,',',time_percent,',',distance_fly,',',classify


'''使用matplotlib绘制数据的散点图'''
# ~ fig = plt.figure()
'''参数111的含义：分为1行1列块画布，并将图像在从左到右从上到下的第1块绘制'''
# ~ ax = fig.add_subplot(111)
# ~ ax.scatter(datingDataMat[:,1],datingDataMat[:,2],20*array(datingLabels),20*array(datingLabels))
# ~ plt.show()


'''归一化特征值'''
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals


'''利用测试集测试错误率'''
def datingClassTest():
    hoRatio = 0.3
    datingDataMat, datingLabels = file2matrix('YHSJ.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestSet = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestSet):
        classifyResult = classify0(normMat[i,:],normMat[numTestSet:m,:],datingLabels[numTestSet:m],11)
        print 'the result come back is:%d,the real result is:%d'%(classifyResult,datingLabels[i])
        if classifyResult != datingLabels[i]:
            errorCount += 1
    print 'the ratio of wrong is:%f'%(errorCount/float(numTestSet))


'''将图像矩阵转化为向量'''
def img2vect(filename):
    returnVect = zeros((0,1024))
    f = open(filename)
    for i in range(32):
        linestr = f.readlines()
        for j in range(32):
            returnVect[0,32*i+j] = int(linestr[j])
     return returnVect


'''测试手写数字0-9的算法'''
def handwritingClassTest():
    hwLabels = []
    filename = raw_input('please input the filename of trainingSet')
    trainingFileStrList = liststr(filename)
    m = len(trainingFileStrList)
    trainingSet = zeros((m,1024))
    for i in range(m):
        trainingFileStr = trainingFileStrList[i].split('.')[0]
        hwLabel = int(trainingFileStr.split('_')[0])
        hwLabels.append(hwLabel)
        trainingSet[i,:] = img2vect('filename:%s'%trainingFileStrList[i])
    filename2 = raw_input('please input the filename of trainingSet')
    testFileStrList = liststr(filename2)
    mtest = len(testFileStrList)
    trainingSet = zeros((m,1024)) 
    errorCount = 0.0
    for i in range(mtest):
        testFileStr = teatFileStrList[i].split('.')[0]
        hwlabel = testFileStr.split('_')[0]
        testVector = img2vect('filename2:%s'%teatFileStrList[i])
        testLabel = classify0(testVector,testFileStrList,hwLabels,5)
        print 'the result come back is:%d,the real result is:%d'%(testLabel,hwlabel)
        if testLabel != hwlabel:
            errorCount += 1
        print 'the ratio of wrong is:%f'%(errorCount/float(mtest))
