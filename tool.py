import pandas as pd

def readUserData(path=''):
    '''
    读取用户数据并存储为csv文件
    :param path: 文件路径
    :return: DataFrame
    '''
    f = pd.read_table(path,sep='::',names=['userID','Gender','Age','Occupation','Zip-code'])
    f.to_csv('users.csv',index=False)
    return f

def readRatingDate(path=''):
    '''
    读取评分数据并存储为csv文件
    :param path:文件路径
    :return: DataFrame
    '''
    f = pd.read_table(path,sep='::',names=['UserID','MovieID','Rating','Timestamp'])
    f.to_csv('ratings.csv',index=False)
    return f

def readMoviesData(path=''):
    '''
    读取电影数据并存储为csv文件
    :param path: 文件路径
    :return: DataFrame
    '''
    f = pd.read_table(path,sep='::',names=['MovieID','Title','Genres'])
    f.to_csv('movies.csv',index=False)
    return f
