# -*- coding: utf-8 -*-
'''
Created on Sat Jul  8  2017
@author: xu jianjie
该程序为用逻辑回归预测薪资概率，并提交最终csv文件
'''
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

#二值化收入和性别,国籍
def encode_income(x): return 1 if x == '>50K' else 0
def encode_sex(x): return 1 if x == 'Male' else 0
def encode_native_country(x): return 1 if x == 'United-States' else 0

#数据标准化
def data_normalization(data):
    if hasattr(data,'income'):data['income'] = data.income.apply(encode_income)
    data['sex'] = data.sex.apply(encode_sex)
    data['native_country'] = data.native_country.apply(encode_native_country)
    
    temp1 = data.select_dtypes(include=[np.number])#提取data数字部分
    categoricals = data.select_dtypes(exclude=[np.number])#提取data非数字部分
    temp2 = pd.get_dummies(categoricals)#非数字特征进行数字化
    features = pd.merge(temp1,temp2,sort=False,copy=False,left_index=True,right_index=True)#合并
    return features


#读训练数据
train = pd.read_csv('./dataset/income/income_train.csv')
#读测试数据
test = pd.read_csv('./dataset/income/income_test.csv')


#抛弃一些特征
train = train.drop(['fnlwgt','education_num'], axis=1)
test = test.drop(['fnlwgt','education_num'], axis=1)


#收入的训练集
y_train = train.income
#特征训练集
x_train =  data_normalization(train.drop(['income','id'],axis=1))
#特征测试集
x_test = data_normalization(test.drop(['id'],axis=1))

#建立逻辑回归模型
lr = LogisticRegression()
#拟合
model=lr.fit(x_train,y_train)
#预测
predictions = model.predict_proba(x_test)


#提交数据csv
submission =pd.DataFrame()
submission['id']=test.id#id部分
submission['income_prob']=predictions[:,1]#概率部分
submission.to_csv('income_result.csv',index=False)   