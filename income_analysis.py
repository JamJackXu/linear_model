# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8  2017
@author: xu jianjie
本程序用于分析income_train.csv和income_test.csv数据集，
并用train数据集尝试训练建立逻辑回归模型，根据损失函数评价模型效果。
"""
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
data1 = pd.read_csv('./dataset/income/income_train.csv')
data2 = pd.read_csv('./dataset/income/income_test.csv')

#检查train和test数据集的空白或缺失值
nulls1 = pd.DataFrame(data1.isnull().sum().sort_values(ascending=False))
nulls2 = pd.DataFrame(data2.isnull().sum().sort_values(ascending=False))

#查看train和test数据的非数字特征
categoricals1 = data1.select_dtypes(exclude=[np.number])
print(categoricals1.describe())
categoricals2 = data2.select_dtypes(exclude=[np.number])
print(categoricals2.describe())

np.set_printoptions(threshold=np.inf)#当数据多时，在控制台打印全部值
for i in range(len(categoricals1.columns)):
    print('i:\n',i)
    print(categoricals1.icol(i).unique())#对每一类打印独立的值

#变换y值为数字特征
def encode_income(x): return 1 if x == '>50K' else 0
data1['income'] = data1.income.apply(encode_income)

def encode_sex(x): return 1 if x == 'Male' else 0
data1['sex'] = data1.sex.apply(encode_sex)

#def encode_workclass(x): return 'Private' if x == '?' else x
#data['workclass'] = data.workclass.apply(encode_workclass)
#
#def encode_occupation(x): return 'Exec-managerial' if x == '?' else x
#data['occupation'] = data.occupation.apply(encode_occupation)

#def encode_race(x): return 1 if x == 'White' else 0
#data['race'] = data.race.apply(encode_race)
#
def encode_native_country(x): return 1 if x == 'United-States' else 0
data1['native_country'] = data1.native_country.apply(encode_native_country)

#抛弃部分特征
data1 = data1.drop(['fnlwgt','education_num'], axis=1)

#def encode_native_country(x): return 'United-States' if x == '?' else x
#categoricals['native_country'] = categoricals.native_country.apply(encode_native_country)


#非数字特征进行数字化
temp1 = data1.select_dtypes(include=[np.number])
temp2 = pd.get_dummies(categoricals1)
features = pd.merge(temp1,temp2,sort=False,copy=False,left_index=True,right_index=True)

y = features.income
x = features.drop(['income','id'], axis=1)
#随机划分训练数据集，1/3测试，2/3训练
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=.33)

lr = LogisticRegression()#建立逻辑模型
model = lr.fit(x_train, y_train)#拟合
predictions = model.predict_proba(x_test)#预测

#损失函数
def logloss(act,pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon,pred)
    pred = sp.minimum(1-epsilon,pred)
    ll = sum(act*sp.log(pred)+sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll* -1.0/len(act)
    return ll
result=logloss(y_test,predictions[:,1])
print('logloss is \n',result)#打印损失函数值

