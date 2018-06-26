# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8  2017
@author: xu jianjie
本程序用于分析house_train.csv和house_test.csv数据集，
并用train数据集尝试训练建立线性回归模型，根据损失函数评价模型效果。
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


#读训练数据
data = pd.read_csv('./dataset/house/house_train.csv')
data.price.describe()#查看价格属性
print("skew is:",data.price.skew())#查看价格数据偏斜度

#设置画图属性
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] =(10,6)

#查看价格分布直方图
plt.hist(data.price)
plt.show()

#取对数后的价格偏斜度和直方图
target = np.log(data.price)
print("new skew is:",target.skew())
plt.hist(target)
plt.show()


#日期数字化
date_n = []
for i in range(len(data.date)):
    x = data.date[i]
    xx = x[0:8]
    date_n.append(float(xx))
data['date'] = date_n

    
features = data.select_dtypes(include=[np.number])
features.dtypes
corr = features.corr()
print (corr['price'].sort_values(ascending=False), '\n')



#画出各个变量与价格的散点图
for i in ['date','bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront',
          'view','condition','grade','sqft_above','sqft_basement','yr_built',
          'yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15']:
    plt.scatter(x=data[i],y=target)
    plt.xlabel(i)
    plt.show()

#根据散点图，抛弃部分奇异值
data = data[data['bedrooms']<10]
data = data[data['bathrooms']<7]
data = data[data['sqft_living']<8000]#8000
data = data[data['sqft_lot']<500000]#500000
data = data[data['sqft_above']<6000]#6000
data = data[data['sqft_basement']<3000]#3000 
data = data[data['long']<-121.6] 
data = data[data['sqft_living15']<5000]
data = data[data['sqft_lot15']<400000]


y = np.log(data.price)
x = data.drop(['price','id'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=.33)

#建立线性模型
lr = linear_model.LinearRegression()
#拟合
model = lr.fit(x_train, y_train)
#预测
predictions = model.predict(x_test)



actual_values = y_test
plt.scatter(predictions, actual_values, alpha=.75, color='b')
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Linear Regression Model')
plt.show()
#均方误差，评价模型预测效果，值越小越好
print ('RMSE is: \n', mean_squared_error(y_test, predictions))

    
#正则化项改进分析
for i in range(-2, 2):
    alpha = 10**i
    rm = linear_model.Ridge(alpha=alpha)
    ridge_model = rm.fit(x_train, y_train)
    preds_ridge = ridge_model.predict(x_test)
    
    plt.scatter(preds_ridge, actual_values, alpha=.75, color='b')
    plt.xlabel('Predicted Price')
    plt.ylabel('Actual Price')
    plt.title('Ridge Regularization with alpha = {}'.format(alpha))
    plt.show()
    
    print('RMSE is:',mean_squared_error(y_test, preds_ridge))
    