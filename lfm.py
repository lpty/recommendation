from multiprocessing import Pool, Manager
from math import exp
import pandas as pd
import numpy as np
import pickle
import time


def getResource(csvPath):
    '''
    获取原始数据
    :param csvPath: csv原始数据路径
    :return: frame
    '''
    frame = pd.read_csv(csvPath)
    return frame


def getUserNegativeItem(frame, userID):
    '''
    获取用户负反馈物品：热门但是用户没有进行过评分 与正反馈数量相等
    :param frame: ratings数据
    :param userID:用户ID
    :return: 负反馈物品
    '''
    userItemlist = list(set(frame[frame['UserID'] == userID]['MovieID']))                       #用户评分过的物品
    otherItemList = [item for item in set(frame['MovieID'].values) if item not in userItemlist] #用户没有评分的物品
    itemCount = [len(frame[frame['MovieID'] == item]['UserID']) for item in otherItemList]      #物品热门程度
    series = pd.Series(itemCount, index=otherItemList)
    series = series.sort_values(ascending=False)[:len(userItemlist)]                            #获取正反馈物品数量的负反馈物品
    negativeItemList = list(series.index)
    return negativeItemList


def getUserPositiveItem(frame, userID):
    '''
    获取用户正反馈物品：用户评分过的物品
    :param frame: ratings数据
    :param userID: 用户ID
    :return: 正反馈物品
    '''
    series = frame[frame['UserID'] == userID]['MovieID']
    positiveItemList = list(series.values)
    return positiveItemList


def initUserItem(frame, userID=1):
    '''
    初始化用户正负反馈物品,正反馈标签为1,负反馈为0
    :param frame: ratings数据
    :param userID: 用户ID
    :return: 正负反馈物品字典
    '''
    positiveItem = getUserPositiveItem(frame, userID)
    negativeItem = getUserNegativeItem(frame, userID)
    itemDict = {}
    for item in positiveItem: itemDict[item] = 1
    for item in negativeItem: itemDict[item] = 0
    return itemDict


def initPara(userID, itemID, classCount):
    '''
    初始化参数q,p矩阵, 随机
    :param userCount:用户ID
    :param itemCount:物品ID
    :param classCount: 隐类数量
    :return: 参数p,q
    '''
    arrayp = np.random.rand(len(userID), classCount)
    arrayq = np.random.rand(classCount, len(itemID))
    p = pd.DataFrame(arrayp, columns=range(0,classCount), index=userID)
    q = pd.DataFrame(arrayq, columns=itemID, index=range(0,classCount))
    return p,q


def work(id, queue):
    '''
    多进程slave函数
    :param id: 用户ID
    :param queue: 队列
    '''
    print(id)
    itemDict = initUserItem(frame, userID=id)
    queue.put({id:itemDict})


def initUserItemPool(userID):
    '''
    初始化目标用户样本
    :param userID:目标用户
    :return:
    '''
    pool = Pool()
    userItem = []
    queue = Manager().Queue()
    for id in userID: pool.apply_async(work, args=(id,queue))
    pool.close()
    pool.join()
    while not queue.empty(): userItem.append(queue.get())
    return userItem


def initModel(frame, classCount):
    '''
    初始化模型：参数p,q,样本数据
    :param frame: 源数据
    :param classCount: 隐类数量
    :return:
    '''
    userID = list(set(frame['UserID'].values))
    itemID = list(set(frame['MovieID'].values))
    p, q = initPara(userID, itemID, classCount)
    userItem = initUserItemPool(userID)
    return p, q, userItem


def sigmod(x):
    '''
    单位阶跃函数,将兴趣度限定在[0,1]范围内
    :param x: 兴趣度
    :return: 兴趣度
    '''
    y = 1.0/(1+exp(-x))
    return y


def lfmPredict(p, q, userID, itemID):
    '''
    利用参数p,q预测目标用户对目标物品的兴趣度
    :param p: 用户兴趣和隐类的关系
    :param q: 隐类和物品的关系
    :param userID: 目标用户
    :param itemID: 目标物品
    :return: 预测兴趣度
    '''
    p = np.mat(p.ix[userID].values)
    q = np.mat(q[itemID].values).T
    r = (p * q).sum()
    r = sigmod(r)
    return r


def latenFactorModel(frame, classCount, iterCount, alpha, lamda):
    '''
    隐语义模型计算参数p,q
    :param frame: 源数据
    :param classCount: 隐类数量
    :param iterCount: 迭代次数
    :param alpha: 步长
    :param lamda: 正则化参数
    :return: 参数p,q
    '''
    p, q, userItem = initModel(frame, classCount)
    for step in range(0, iterCount):
        for user in userItem:
            for userID, samples in user.items():
                for itemID, rui in samples.items():
                    eui = rui - lfmPredict(p, q, userID, itemID)
                    for f in range(0, classCount):
                        print('step %d user %d class %d' % (step, userID, f))
                        p[f][userID] += alpha * (eui * q[itemID][f] - lamda * p[f][userID])
                        q[itemID][f] += alpha * (eui * p[f][userID] - lamda * q[itemID][f])
        alpha *= 0.9
    return p, q


def recommend(frame, userID, p, q, TopN=10):
    '''
    推荐TopN个物品给目标用户
    :param frame: 源数据
    :param userID: 目标用户
    :param p: 用户兴趣和隐类的关系
    :param q: 隐类和物品的关系
    :param TopN: 推荐数量
    :return: 推荐物品
    '''
    userItemlist = list(set(frame[frame['UserID'] == userID]['MovieID']))
    otherItemList = [item for item in set(frame['MovieID'].values) if item not in userItemlist]
    predictList = [lfmPredict(p, q, userID, itemID) for itemID in otherItemList]
    series = pd.Series(predictList, index=otherItemList)
    series = series.sort_values(ascending=False)[:TopN]
    return series


if __name__ == '__main__':
    frame = getResource('csvResource/ratings.csv')
    p, q = latenFactorModel(frame, 5, 10, 0.02, 0.01)
    l = recommend(frame, 1, p, q)
    print(l)

