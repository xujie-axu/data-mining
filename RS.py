# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 16:34:56 2017

@author: yunhe
"""
import numpy as np
import heapq
from scipy.sparse import lil_matrix
import math

def Loadtraindata(fileName='train.txt'):
    str1 = './'                         # 目录的相对地址    
    UserData = {}
    global totalitem
    totalitem=[]
    username=0
    userNum=0
    
    for line in open(str1+fileName,'r'):
        if( '|' in line):
            userNum += 1           
            UId = line[0:line.find('|')]
            username=UId
        else:           
            a=line.split('  ')
            if len(a)>1:
                item=a[0]
                totalitem.append(int(item))
                rating=a[1]
                UserData.setdefault(username, {})      # 设置字典的默认格式,元素是user:{}字典
                UserData[username][item] = float(rating)
            else:
                break
    totalitem=list(set(totalitem))#商品编号列表，已排序
    print('加载结束')
    return (UserData)

def Loadtestdata(fileName='test.txt'):
    str1 = './'                         # 目录的相对地址    
    UserData = {}
    global totalitem
    username=0
    userNum=0    
    for line in open(str1+fileName,'r'):
        if( '|' in line):
            userNum += 1           
            UId = line[0:line.find('|')]
            username=UId
        else:  
            a=line.split('\n')
            item=a[0]
            UserData.setdefault(username, {})      # 设置字典的默认格式,元素是user:{}字典
            UserData[username][item] = 0
    print('加载结束')
    return (UserData)

#生成U-V矩阵-用户评分矩阵
def UV_RatingMatrix(data):
    l = lil_matrix((19835,max(totalitem)+1))#用户编号是连续的，商品编号不连续
    for userId in data:
        for itemId in data[userId]:
            l[int(userId),int(itemId)]=data[userId][itemId]
    return l

#for 训练集；生成U-V矩阵-购买记录矩阵
def UV_preferMatrix():
    l = lil_matrix((19835,max(totalitem)+1))#用户编号是连续的，商品编号不连续
    for userId in traindata:
        for itemId in traindata[userId]:
            if traindata[userId][itemId]!=0:
                l[int(userId),int(itemId)]=1
    l=l.astype('int32')
    return l
#获取某一用户相似的top100个用户
def Neighbor_of(user):
    k=(user).toarray()
    for data in k:
        neighbors=np.argsort(-data)[0:100]#取相似用户top100
    return neighbors
#生成用户关系字典
def UUMatrix():
    uu=lprefer_t*lprefer_t.T
    neighbor={}     
    for x in testdata:
        user=uu[int(x)]
        neighborId=Neighbor_of(user)
        neighbor.setdefault(x, [])      # 设置字典的默认格式,元素是user:{}字典
        neighbor[x] = list(neighborId)
    return neighbor

# =============================================================================
# part2
# 依据用户关系，推荐评分  
# output pearson correlation score  
def sim_pearson(traindata, userId, neighborId):
    sim = {}
 
    #查找双方都评价过的项
    for item in traindata[userId]:
        if item in traindata[neighborId]:
            sim[item] = 1           #将相同项添加到字典sim中
    #元素个数
    n = len(sim)
    if len(sim)==0:
        return -1

    # 所有偏好之和
    sum1 = sum([traindata[userId][item] for item in sim])  
    sum2 = sum([traindata[neighborId][item] for item in sim])  

    #求平方和
    sum1Sq = sum( [pow(traindata[userId][item] ,2) for item in sim] )
    sum2Sq = sum( [pow(traindata[neighborId][item] ,2) for item in sim] )

    #求乘积之和 ∑XiYi
    sumMulti = sum([traindata[userId][item]*traindata[neighborId][item] for item in sim])

    num1 = sumMulti - (sum1*sum2/n)
    num2 = math.sqrt( (sum1Sq-pow(sum1,2)/n)*(sum2Sq-pow(sum2,2)/n))  
    if num2==0: 
        return 0
       ### 如果分母为0，本处将返回0.
    else:
        result = num1/num2
        return result

#获取用户评分平均值
def aveRating(userId):#userId(char)
    rating=0
    for x in traindata[userId]:
        rating+=traindata[userId][x]
    rating=rating/len(traindata[userId])
    return rating
#获取商品的平均得分
def aveitemRating(itemId):
    score=0
    count=0
    for data in lRating_T[int(itemId)].data:
        if len(data)!=0:
            score=sum(x for x in data)/len(data)
        else:
            score=0
            print(count)
    return score
def RecommendItemRating(k,userId,itemId,uu):
     
    sltnb=[]
    Pear=[] #[相似度，用户ID]
    Pears=[]
    score=0
    weight=0 #sum(相关度)，用于归一化
    
    neighbors=uu[userId] 
    for neighbor in neighbors:
        if itemId in traindata[str(neighbor)]:
            sltnb.append(neighbor)
    for neighborId in sltnb:
        Pear.append(sim_pearson(traindata,userId,str(neighborId))) 
        Pear.append(neighborId)
        Pears.append(Pear)
        Pear=[]
###### Pear=np.array(Pear)#对相似度排序 ######
    #如果相关用户小于K个，那么他们就是valuable neighbor的全部
    if len(Pears)<k & len(Pears)>0:
        for x in Pears:
            if x[0]>0:
                score += x[0]*round(traindata[x[1]][itemId]-aveRating(x[1]))
                weight += x[0]
        if weight==0:
            score=0
        else:
            score = score/weight # 推荐分数=sum(用户打分偏移量*相似度权值)/sum(相似度权值)     
            rating=round(aveRating(userId)+score)
    #如果没有相关用户，那么推荐分数为商品的平均得分
    elif len(Pears)==0:
        rating = round(aveitemRating(itemId))
    #相关用户取top K
    else:
        topPears=heapq.nlargest(k, Pears)
        for x in topPears:
            if x[0]>0:
                score += x[0]*round(traindata[str(x[1])][itemId]-aveRating(str(x[1])))
                weight += x[0]
        if weight==0:
            score=0
        else:        
            score = score/weight 
        rating=round(aveRating(userId)+score)
    # 修正得分边界
    if(rating<0):
        rating=0
    if(rating>100):
        rating=100
    return rating
        
# =============================================================================
if __name__ == "__main__":
    # load data
    traindata = Loadtraindata()
    testdata = Loadtestdata()    
    usernum=len(testdata)
    print (len(traindata))
    print (len(testdata))
    print (""" 生成成功 """)
    # create UV matrix
    lRating=UV_RatingMatrix(traindata) 
    lRating_T=lRating.T
    lprefer_t=UV_preferMatrix() 
    
    #compute recommended rating
    neighbor=UUMatrix()
    k=20
    Rpredict={}
    for userId in testdata:
        for itemId in testdata[userId]:
            rating=RecommendItemRating(20,userId,itemId,neighbor)
            Rpredict.setdefault(userId, {}) # 设置字典的默认格式,元素是user:{}字典
            Rpredict[userId][itemId] = float(rating)    
            
    #make result file
    f1=open('test_result3.txt', 'w',encoding='utf-8')
    for x in Rpredict:
        f1.write(x+'|'+str(len(Rpredict[x])))
        f1.write("\n")
        for y in Rpredict[x]:
            f1.write(y)
            f1.write('\t')
            f1.write(str(Rpredict[x][y]))
            f1.write('\n')
    f1.close()







