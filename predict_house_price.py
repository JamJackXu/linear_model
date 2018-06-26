# -*- coding: utf-8 -*-
'''
Created on Sat Jul  8  2017
@author: xu jianjie
该程序为用线性回归预测房价，并提交最终csv文件
'''
import numpy as np
import pandas as pd
from sklearn import linear_model

#把时间变换为float型
def date_normalization(data):
    temp=[]
    for i in range(len(data.date)):
        x = data.date[i]
        xx = x[0:8]
        temp.append(float(xx))
    data['date'] = temp
    return data
#读训练数据
train = pd.read_csv('./dataset/house/house_train.csv')
train =date_normalization(train)

#抛弃奇异数据
train = train[train['bedrooms']<10]
train = train[train['bathrooms']<7]
train = train[train['sqft_living']<8000]#8000
train = train[train['sqft_lot']<500000]#500000
train = train[train['sqft_above']<6000]#6000
train = train[train['long']<-121.6] 
train = train[train['sqft_living15']<5000]
train = train[train['sqft_lot15']<400000]
train = train[train['sqft_basement']<3000]#3000
#读测试数据
test = pd.read_csv('./dataset/house/house_test.csv')
test =date_normalization(test)
#训练价格取对数
y_train = np.log(train.price)
#抛弃训练特征价格和id
x_train =  train.drop(['price','id'],axis=1)
#抛弃测试特征id
x_test = test.drop(['id'],axis=1)

#建立线性模型
lr=linear_model.LinearRegression()
#拟合
model=lr.fit(x_train,y_train)
#预测
predictions = model.predict(x_test)


#提交数据csv
submission =pd.DataFrame()
submission['id']=test.id#id部分
submission['price']=np.exp(predictions)#房价部分，取指数返回得到真实价格
submission.to_csv('house_result.csv',index=False)   