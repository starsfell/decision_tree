# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 09:26:36 2018

@author: xintong.yan
"""

'''
1. 传统的代码算法实现ID3，C4.5决策树
2. 用sklearn的包建立Entropy与Gini决策树

'''

################################ 传统方法 ID3 实现 #############################
from math import log
import operator
import xlrd
import xlwt
import math
import operator
from datetime import date,datetime
from sklearn import datasets
import numpy as np

'创造实例数据'
def createDataSet1():
    dataSet = [['是','否','OK',  '多云','去'  ],
               ['是','否','OK',  '晴朗','去'  ],
               ['是','是','OK',  '多云','去'  ],
               ['是','否','不OK','晴朗','去'  ],
               ['否','否','OK',  '多云','去'  ],
               ['是','是','OK',  '晴朗','去'  ],
               ['是','否','不OK','多云','去'  ],
               ['是','否','OK',  '下雨','去'  ],
               ['否','是','不OK','下雨','不去'],
               ['否','是','OK',  '下雨','不去'],
               ['否','否','不OK','多云','不去'],
               ['否','否','OK',  '下雨','不去'],
               ['否','是','OK',  '多云','不去'],
               ['是','是','不OK','下雨','不去'],
               ['否','是','OK',  '晴朗','不去'],
               ['否','是','不OK','多云','不去'],
               ['否','否','不OK','晴朗','不去']]
    labels = ['SXR','JB','NPY','TQ']
    return dataSet, labels

    
'计算数据的总熵值'
'@ 输入： 数据集dataSet'
'@ 输出： 数据集整体的信息熵Entropy'
def calcShannonEnt(dataSet):            ## 定义计算数据的熵Entropy的公式
    numEntries = len(dataSet)              ## 计算数据条数，共17条数据
    labelCounts={}                         ## 定义一个字典
    for featVec in dataSet:                ## 数据集中的每一条数据就对应一条featVec
        currentLabel=featVec[-1]           ## 每行数据的最后一个字段取值，就是Y的值提出来
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0      ## 逐渐累积每类的个数
        labelCounts[currentLabel]+=1       ##统计有多少个类，以及每个类的数量
    shannonEnt=0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries   ## 计算每个类的熵
        shannonEnt -= prob * log(prob,2)          ## 累加每个类的熵
    return shannonEnt

    
'按某个特征分类后的数据'
'划分数据集, 提取所有满足一个特征的值'
'@ 输入： dataSet: 数据集'
'@ 输入： axis: 划分数据集的特征,axis可以取0，1，2，3，这4列'
'@ 输入： value提取出来满足某特征的list，value是每个自变量能取的值'
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:        ## 将相同数据特征的提取出来
        if featVec[axis]==value:   ## axis可以选择0，1，2，3；就是数据集的第一列值、第二列值，第三列值，第四列值，对应4列自变量。
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])  #将Y值加上
            retDataSet.append(reducedFeatVec)
    return retDataSet


 
'选择最优分类特征,第一个分支的选择算法'
'@ 输入： dataSet: 数据集'
'@ 输出： bestFeature：最佳划分属性'
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0])-1           ## 属性个数
    baseEntropy = calcShannonEnt(dataSet)     ## 数据集的熵
    bestInfoGain = 0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]   # 提取数据集中自变量的每列数据
        uniqueVals = set(featList)           # 每个自变的可能取值 distinct这一列
        newEntropy = 0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)   ##按特征分类后的熵
        infoGain = baseEntropy - newEntropy    # 原始熵与按特征分类后的熵的差值
        if (infoGain>bestInfoGain):            # 若按某特征划分后，熵值减少的最大，则该特征为最优分类特征
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
 
    
'按分类后类别数量排序，比如：最后分类为2去1不去，则判定为去'
'递归构建决策树'
'@ 输入：classList: 类别列表'
'@ 输出：sortedClassCount[0][0]: 出现次数最多的类别'

def majorityCnt(classList): 
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]     # 返回出现次数最多的因变量结果


'构造决策树'
'@ 输入：dataSet: 数据集'
'@ 输入：labels: 标签集'
'@ 输出：myTree: 决策树'

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]     # 类别去或不去
    if classList.count(classList[0]) == len(classList):  # 当类别与属性完全相同时停止
        return classList[0]
    if len(dataSet[0]) == 1:                           # 遍历完所有特征值时，返回数量最多的
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)     ## 选择最优特征
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}  # 分类结果以字典保存
    del(labels[bestFeat])        # 清空labels[bestFeat]
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat,value), subLabels)  # 递归调用创建决策树
    return myTree
     
    
if __name__ == '__main__':
    dataSet, labels = createDataSet1()    # 创造示例数据
    print(createTree(dataSet,labels))     # 输出决策树结果


## 绘图
def getNumLeafs(myTree):
    '计算决策树的叶子数'
    # 叶子数
    numLeafs = 0
    # 节点信息
    sides = list(myTree.keys())  
    firstStr =sides[0]
    # 分支信息
    secondDict = myTree[firstStr]
    for key in secondDict.keys():   # 遍历所有分支
        # 子树分支则递归计算
        if type(secondDict[key]).__name__=='dict':
            numLeafs += getNumLeafs(secondDict[key])
        # 叶子分支则叶子数+1
        else:   numLeafs +=1  
    return numLeafs	

    
def getTreeDepth(myTree):
    '计算决策树的深度'
    # 最大深度
    maxDepth = 0
    # 节点信息
    sides = list(myTree.keys())   
    firstStr =sides[0]
    # 分支信息
    secondDict = myTree[firstStr]
    for key in secondDict.keys():   # 遍历所有分支
        # 子树分支则递归计算
        if type(secondDict[key]).__name__=='dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        # 叶子分支则叶子数+1
        else:   thisDepth = 1
        # 更新最大深度
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

## 绘图
import matplotlib.pyplot as plt
 
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

# ==================================================
# 输入：
#        nodeTxt:     终端节点显示内容
#        centerPt:    终端节点坐标
#        parentPt:    起始节点坐标
#        nodeType:    终端节点样式
# 输出：
#        在图形界面中显示输入参数指定样式的线段(终端带节点)
# ==================================================
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    '画线(末端带一个点)'   
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction', va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )
    
# =================================================================
# 输入：
#        cntrPt:      终端节点坐标
#        parentPt:    起始节点坐标
#        txtString:   待显示文本内容
# 输出：
#        在图形界面指定位置(cntrPt和parentPt中间)显示文本内容(txtString)
# =================================================================
def plotMidText(cntrPt, parentPt, txtString):
    '在指定位置添加文本'
    # 中间位置坐标
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)
 
# ===================================
# 输入：
#        myTree:    决策树
#        parentPt:  根节点坐标
#        nodeTxt:   根节点坐标信息
# 输出：
#        在图形界面绘制决策树
# ===================================
def plotTree(myTree, parentPt, nodeTxt):
    '绘制决策树'
    
    # 当前树的叶子数
    numLeafs = getNumLeafs(myTree)
    # 当前树的节点信息
    sides = list(myTree.keys())   
    firstStr =sides[0]
    
    # 定位第一棵子树的位置(这是蛋疼的一部分)
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    
    # 绘制当前节点到子树节点(含子树节点)的信息
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    
    # 获取子树信息
    secondDict = myTree[firstStr]
    # 开始绘制子树，纵坐标-1。        
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
      
    for key in secondDict.keys():   # 遍历所有分支
        # 子树分支则递归
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key],cntrPt,str(key))
        # 叶子分支则直接绘制
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
     
    # 子树绘制完毕，纵坐标+1。
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
 
# ==============================
# 输入：
#        myTree:    决策树
# 输出：
#        在图形界面显示决策树
# ==============================
def createPlot(inTree):
    '显示决策树'
    
    # 创建新的图像并清空 - 无横纵坐标
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    
    # 树的总宽度 高度
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    
    # 当前绘制节点的坐标
    plotTree.xOff = -0.5/plotTree.totalW; 
    plotTree.yOff = 1.0;
    
    # 绘制决策树
    plotTree(inTree, (0.5,1.0), '')
    
    plt.show()
        

if __name__ == '__main__':
    dataSet, labels = createDataSet1()    # 创造示例数据
    split_fea_index=chooseBestFeatureToSplit(dataSet)
    myTree = createTree(dataSet,labels)
    print(myTree)     # 输出决策树结果
    createPlot(myTree)
	

################################ 传统方法 C4.5实现 #############################
#coding=utf-8
import xlrd
import xlwt
import math
import operator
from datetime import date,datetime
from sklearn import datasets
 
#导入数据
def createDataSet1():
    dataSet = pd.read_table('C:/Users/xintong.yan/Desktop/sample_data.txt',encoding='GBK')
    labels = list(dataSet.columns.values)[1:]  # 数据的列名，自变量与因变量，[1：]是为了去除编号这个变量
    dataSet = dataSet.ix[:,labels]       # 把编号这一列从数据里剔除
    dataSet = dataSet.values.tolist()   # 把dataframe转成list
    return dataSet, labels   
    
##计算给定数据集的信息熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():     #为所有可能分类创建字典
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * math.log(prob,2)   #以2为底数求对数
    return shannonEnt

'''
#创建数据
def createDataSet1():
    dataSet = [['是','否','OK',  '多云','去'  ],
               ['是','否','OK',  '晴朗','去'  ],
               ['是','是','OK',  '多云','去'  ],
               ['是','否','不OK','晴朗','去'  ],
               ['否','否','OK',  '多云','去'  ],
               ['是','是','OK',  '晴朗','去'  ],
               ['是','否','不OK','多云','去'  ],
               ['是','否','OK',  '下雨','去'  ],
               ['否','是','不OK','下雨','不去'],
               ['否','是','OK',  '下雨','不去'],
               ['否','否','不OK','多云','不去'],
               ['否','否','OK',  '下雨','不去'],
               ['否','是','OK',  '多云','不去'],
               ['是','是','不OK','下雨','不去'],
               ['否','是','OK',  '晴朗','不去'],
               ['否','是','不OK','多云','不去'],
               ['否','否','不OK','晴朗','不去']]
    labels = ['SXR','JB','NPY','TQ']
    return dataSet, labels
'''

 
#依据特征划分数据集  axis代表第几个特征  value代表该特征所对应的值  返回的是划分后的数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
 
 
#选择最好的数据集(特征)划分方式  返回最佳特征下标
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1   #特征个数
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGainrate = 0.0; bestFeature = -1
    for i in range(numFeatures):   #遍历特征 第i个
        featureSet = set([example[i] for example in dataSet])   #第i个特征取值集合
        newEntropy= 0.0
        splitinfo= 0.0
        for value in featureSet:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)   #该特征划分所对应的entropy
            splitinfo -= prob*math.log(prob,2)
        if not splitinfo:
            splitinfo=-0.99*math.log(0.99,2)-0.01*math.log(0.01,2)
        infoGain = baseEntropy - newEntropy
        infoGainrate = float(infoGain)/float(splitinfo)
        if infoGainrate > bestInfoGainrate:
            bestInfoGainrate = infoGainrate
            bestFeature = i
    return bestFeature
 
#创建树的函数代码   python中用字典类型来存储树的结构 返回的结果是myTree-字典
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):    #类别完全相同则停止继续划分  返回类标签-叶子节点
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)       #遍历完所有的特征时返回出现次数最多的
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]    #得到的列表包含所有的属性值
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree
 
#多数表决的方法决定叶子节点的分类 ----  当所有的特征全部用完时仍属于多类
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0;
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)  #排序函数 operator中的
    return sortedClassCount[0][0]
 
#使用决策树执行分类
def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)   #index方法查找当前列表中第一个匹配firstStr变量的元素的索引
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else: classLabel = secondDict[key]
    return classLabel

if __name__ == '__main__':
    dataSet, labels = createDataSet1()    # 创造示例数据
    print(createTree(dataSet,labels))      
    


## 绘图
def getNumLeafs(myTree):
    '计算决策树的叶子数'
    # 叶子数
    numLeafs = 0
    # 节点信息
    sides = list(myTree.keys())  
    firstStr =sides[0]
    # 分支信息
    secondDict = myTree[firstStr]
    for key in secondDict.keys():   # 遍历所有分支
        # 子树分支则递归计算
        if type(secondDict[key]).__name__=='dict':
            numLeafs += getNumLeafs(secondDict[key])
        # 叶子分支则叶子数+1
        else:   numLeafs +=1  
    return numLeafs	

    
def getTreeDepth(myTree):
    '计算决策树的深度'
    # 最大深度
    maxDepth = 0
    # 节点信息
    sides = list(myTree.keys())   
    firstStr =sides[0]
    # 分支信息
    secondDict = myTree[firstStr]
    for key in secondDict.keys():   # 遍历所有分支
        # 子树分支则递归计算
        if type(secondDict[key]).__name__=='dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        # 叶子分支则叶子数+1
        else:   thisDepth = 1
        # 更新最大深度
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

## 绘图
import matplotlib.pyplot as plt
 
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

# ==================================================
# 输入：
#        nodeTxt:     终端节点显示内容
#        centerPt:    终端节点坐标
#        parentPt:    起始节点坐标
#        nodeType:    终端节点样式
# 输出：
#        在图形界面中显示输入参数指定样式的线段(终端带节点)
# ==================================================
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    '画线(末端带一个点)'   
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction', va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )
    
# =================================================================
# 输入：
#        cntrPt:      终端节点坐标
#        parentPt:    起始节点坐标
#        txtString:   待显示文本内容
# 输出：
#        在图形界面指定位置(cntrPt和parentPt中间)显示文本内容(txtString)
# =================================================================
def plotMidText(cntrPt, parentPt, txtString):
    '在指定位置添加文本'
    # 中间位置坐标
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)
 
# ===================================
# 输入：
#        myTree:    决策树
#        parentPt:  根节点坐标
#        nodeTxt:   根节点坐标信息
# 输出：
#        在图形界面绘制决策树
# ===================================
def plotTree(myTree, parentPt, nodeTxt):
    '绘制决策树'
    
    # 当前树的叶子数
    numLeafs = getNumLeafs(myTree)
    # 当前树的节点信息
    sides = list(myTree.keys())   
    firstStr =sides[0]
    
    # 定位第一棵子树的位置(这是蛋疼的一部分)
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    
    # 绘制当前节点到子树节点(含子树节点)的信息
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    
    # 获取子树信息
    secondDict = myTree[firstStr]
    # 开始绘制子树，纵坐标-1。        
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
      
    for key in secondDict.keys():   # 遍历所有分支
        # 子树分支则递归
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key],cntrPt,str(key))
        # 叶子分支则直接绘制
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
     
    # 子树绘制完毕，纵坐标+1。
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
 
# ==============================
# 输入：
#        myTree:    决策树
# 输出：
#        在图形界面显示决策树
# ==============================
def createPlot(inTree):
    '显示决策树'
    
    # 创建新的图像并清空 - 无横纵坐标
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    
    # 树的总宽度 高度
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    
    # 当前绘制节点的坐标
    plotTree.xOff = -0.5/plotTree.totalW; 
    plotTree.yOff = 1.0;
    
    # 绘制决策树
    plotTree(inTree, (0.5,1.0), '')
    
    plt.show()
        

if __name__ == '__main__':
    dataSet, labels = createDataSet1()    # 创造示例数据
    split_fea_index=chooseBestFeatureToSplit(dataSet)
    myTree = createTree(dataSet,labels)
    print(myTree)     # 输出决策树结果
    createPlot(myTree)
	



    
###############################################################################
################################ sklearn ID3实现 #############################

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
import pydotplus
from sklearn.externals.six import StringIO


dataSet = pd.read_csv('C:/Users/xintong.yan/Desktop/sample_data2.csv') 

x = dataSet[[1,2,3,4]]
y = dataSet['Result']


# 拆分数据集和训练集,测试集比重40%
np.random.seed(12)   # set seed,保证抽样结果不变
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4)  ## 随机抽样


# 使用信息熵作为划分标准（information gain）
clf = tree.DecisionTreeClassifier(criterion='entropy')
print(clf)
clf_fit = clf.fit(x_train,y_train)


# 打印特征在分类起到的作用性
print(clf.feature_importances_)


# 打印测试结果
answer = clf.predict(x_train)
print(answer)        ## 用模型预测出的Y hat
print(y_train)       ## 原本的Y hat
print(np.mean(answer==y_train))

# 准确率与召回率
precision,recall,thresholds = precision_recall_curve(y_train,clf.predict(x_train))
answer = clf.predict_proba(x)[:,1]
print(classification_report(y,answer,target_names=['不去','去']))  # 0是不去，1是去

# 导出Tree为dot格式
# 输入为默认路径“C:\Users\xintong.yan”
tree.export_graphviz(clf,out_file='tree.dot')  

## 可视化
feature_name = ['SXR','JB','NPY','TQ']
target_name = ['Q','BQ']

dot_data = StringIO()
tree.export_graphviz(clf,out_file = dot_data,feature_names=feature_name,
                     class_names=target_name,filled=True,rounded=True,
                     special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("Decision_tree-ID3.pdf")
print('Visible tree plot saved as pdf.')





    
###############################################################################
################################ sklearn CART实现 #############################

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
import pydotplus
from sklearn.externals.six import StringIO


dataSet = pd.read_csv('C:/Users/xintong.yan/Desktop/sample_data2.csv') 

x = dataSet[[1,2,3,4]]
y = dataSet['Result']


# 拆分数据集和训练集,测试集比重40%
np.random.seed(12)   # set seed,保证抽样结果不变
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4)  ## 随机抽样


# 使用基尼系数作为划分标准（gini index）
clf = tree.DecisionTreeClassifier(criterion='gini')
print(clf)
clf_fit = clf.fit(x_train,y_train)


# 打印特征在分类起到的作用性
print(clf.feature_importances_)


# 打印测试结果
answer = clf.predict(x_train)
print(answer)        ## 用模型预测出的Y hat
print(y_train)       ## 原本的Y hat
print(np.mean(answer==y_train))

# 准确率与召回率
precision,recall,thresholds = precision_recall_curve(y_train,clf.predict(x_train))
answer = clf.predict_proba(x)[:,1]
print(classification_report(y,answer,target_names=['不去','去']))  # 0是不去，1是去

# 导出Tree为dot格式
# 输入为默认路径“C:\Users\xintong.yan”
tree.export_graphviz(clf,out_file='tree.dot')  

## 可视化
feature_name = ['SXR','JB','NPY','TQ']
target_name = ['Q','BQ']

dot_data = StringIO()
tree.export_graphviz(clf,out_file = dot_data,feature_names=feature_name,
                     class_names=target_name,filled=True,rounded=True,
                     special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("Decision_tree-CART.pdf")
print('Visible tree plot saved as pdf.')



'''
如果是使用sklearn库的决策树生成的话，剪枝方法有限，仅仅只能改变其中参数来进行剪枝。
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=7,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=10,
            min_samples_split=20, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best')

 -- criterion: ”gini” or “entropy”(default=”gini”)是计算属性的gini(基尼不纯度)还是entropy(信息增益)，来选择最合适的节点。
 -- splitter: ”best” or “random”(default=”best”)随机选择属性还是选择不纯度最大的属性，建议用默认。
 -- max_features: 选择最适属性时划分的特征不能超过此值。当为整数时，即最大特征数；当为小数时，训练集特征数*小数；
    -- if “auto”, then max_features=sqrt(n_features).
    -- If “sqrt”, thenmax_features=sqrt(n_features).
    -- If “log2”, thenmax_features=log2(n_features).
    -- If None, then max_features=n_features.
 -- max_depth: (default=None)设置树的最大深度，默认为None，这样建树时，会使每一个叶节点只有一个类别，或是达到min_samples_split。
 -- min_samples_split:根据属性划分节点时，每个划分最少的样本数。分裂点的样本个数
 -- min_samples_leaf:叶子节点最少的样本数。
 -- max_leaf_nodes: (default=None)最大的叶子节点数
 -- min_weight_fraction_leaf: (default=0) 叶子节点所需要的最小权值

另外还有以下几种进阶的剪枝方式：
Reduced-Error Pruning(REP,错误率降低剪枝):
 决定是否修剪这个结点有如下步骤组成：
1：删除以此结点为根的子树
2：使其成为叶子结点
3：赋予该结点关联的训练数据的最常见分类
4：当修剪后的树对于验证集合的性能不会比原来的树差时，才真正删除该结点
从底向上的处理结点，删除那些能够最大限度的提高验证集合的精度的结点，直到会降低验证集合精度为止。





