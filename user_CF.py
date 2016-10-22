import math
import time
import pandas as pd

def calcuteSimilar(series1,series2):
    '''
    计算余弦相似度
    :param data1: 数据集1 Series
    :param data2: 数据集2 Series
    :return: 相似度
    '''
    unionLen = len(set(series1) & set(series2))
    if unionLen == 0: return 0.0
    product = len(series1) * len(series2)
    similarity = unionLen / math.sqrt(product)
    return similarity

def calcuteUser(csvpath,targetID=1,TopN=10):
    '''
    计算targetID的用户与其他用户的相似度
    :return:相似度TopN Series
    '''
    frame = pd.read_csv(csvpath)                                                        #读取数据
    targetUser = frame[frame['UserID'] == targetID]['MovieID']                          #目标用户数据
    otherUsersID = [i for i in set(frame['UserID']) if i != targetID]                   #其他用户ID
    otherUsers = [frame[frame['UserID'] == i]['MovieID'] for i in otherUsersID]         #其他用户数据
    similarlist = [calcuteSimilar(targetUser,user) for user in otherUsers]              #计算
    similarSeries = pd.Series(similarlist,index=otherUsersID)                           #Series
    return similarSeries.sort_values()[-TopN:]

def calcuteInterest(frame,similarSeries,targetItemID):
    '''
    计算目标用户对目标物品的感兴趣程度
    :param frame: 数据
    :param similarSeries: 目标用户最相似的K个用户
    :param targetItemID: 目标物品
    :return:感兴趣程度
    '''
    similarUserID = similarSeries.index                                                 #和用户兴趣最相似的K个用户
    similarUsers = [frame[frame['UserID'] == i] for i in similarUserID]                 #K个用户数据
    similarUserValues = similarSeries.values                                            #用户和其他用户的兴趣相似度
    UserInstItem = []
    for u in similarUsers:                                                              #其他用户对物品的感兴趣程度
        if targetItemID in u['MovieID'].values: UserInstItem.append(u[u['MovieID']==targetItemID]['Rating'].values[0])
        else: UserInstItem.append(0)
    interest = sum([similarUserValues[v]*UserInstItem[v]/5 for v in range(len(similarUserValues))])
    return interest

def calcuteItem(csvpath,targetUserID=1,TopN=10):
    '''
    计算推荐给targetUserID的用户的TopN物品
    :param csvpath: 数据路径
    :param targetUserID: 目标用户
    :param TopN:
    :return: TopN个物品及感兴趣程度
    '''
    frame = pd.read_csv(csvpath)                                                        #读取数据
    similarSeries = calcuteUser(csvpath=csvpath, targetID=targetUserID)                 #计算最相似K个用户
    userMovieID = set(frame[frame['UserID'] == 1]['MovieID'])                           #目标用户感兴趣的物品
    otherMovieID = set(frame[frame['UserID'] != 1]['MovieID'])                          #其他用户感兴趣的物品
    movieID = list(userMovieID ^ otherMovieID)                                          #差集
    interestList = [calcuteInterest(frame,similarSeries,movie) for movie in movieID]    #推荐
    interestSeries = pd.Series(interestList, index=movieID)
    return interestSeries.sort_values()[-TopN:]                                         #TopN

if __name__ == '__main__':
    print('start..')
    start = time.time()
    a = calcuteItem('csvResource\\ratings.csv')
    print(a)
    print('Cost time: %f'%(time.time()-start))