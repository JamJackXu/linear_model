# linear_model
### You may see the code elsewhere. Please leave a message, if there exits any conflict of interest.</br>
The house prices are predicted by linear regression and the employee incomes are predicted by logistic regression.

predict_house_price.py：预测房价，生成csv文件</br>
house_analysis.py：分析房价，scores评价模型</br>
house_result.csv：最终生成文件</br>
</br>
predict_employee_income.py：预测薪资，生成csc文件</br>
income_analysis.py：分析薪资数据，logloss评价模型</br>
income_result.csv：最终生成文件</br>
#### I、提供美国某县2014年5月到2015年5月房屋销售价格数据（文件名为house_train.csv），请根据历史销售数据训练模型并预测给定房屋(文件名为house_test.csv)的销售价格，提交预测结果(house_redult.csv)

| 字段名 | 含义 |
| - | :-: | 
| id | 销售事件唯一编号 | 
| date | 销售日期 | 
| price | 销售价格 | 
| Bedrooms | 房间数量 | 
| Bathrooms | 卫生间数量(0.5表示有一个半功能的卫生间) | 
| sqft_living | 起居室内部面积 | 
| sqft_lot | 占地面积 | 
| Floor | 房屋含有的层数 | 
| Waterfront | 是否可以看到水面（能看到江,河,湖,海） | 
| View | 房屋整体视觉效果评分(0-4) | 
| Condition | 房屋的整体条件评分(1-5) | 
| Grade | 房屋结构以及设计评分(1-13) | 
| Sqft_above | 地面以上的居住面积 | 
| sqft_basement | 地下室面积 | 
| yr_build | 房屋建造时间 | 
| yr_renovated | 房屋最近一次翻新的时间 | 
| Zipcode | 房子所在的邮编 | 
| Lat | 地处位置的纬度 | 
| Long | 地处位置的经度 | 
| sqft_living15 | 距离房屋最近的15个房子的起居室面积 | 
| sqft_lot15 | 距离房屋最近的15个房子的占地面积 |
##### 表一 数据集字段解释
| Name | meaning |
| - | :-: | 
| id | 销售事件唯一编号 | 
| price | 销售价格 | 
##### 表二 需要提交的结果house_result.csv的结构</br>
由提交的结果(house_result.csv)根据平均绝对误差(RMSE)计算成绩，评分函数为<img src="http://latex.codecogs.com/gif.latex?score=\sqrt{\frac{\sum_{i=1}^{N}(y_{i}-f_{i})^{2}}{N}}" title="score=\sqrt{\frac{\sum_{i=1}^{N}(y_{i}-f_{i})^{2}}{N}}" />
其中,是预测样本数量，<img src="http://latex.codecogs.com/gif.latex?y_{i}" title="y_{i}" />为样本<img src="http://latex.codecogs.com/gif.latex?i" title="i" />真实房价，<img src="http://latex.codecogs.com/gif.latex?f_{i}" title="f_{i}" />是你根据训练模型预测样本 的房价， 越低越好，越低说明预测越准确。

#### II、提供某地区社会成员薪资状况(文件名为income_train.csv)，根据提供的数据集进行建模，请根据训练好的模型预测给定特征人员(income_test.csv)的薪资情况并提交结果(income_result.csv)，提交结果为对应人员薪资>=50K的概率。本题假定薪资>=50K为正例1，薪资<=50K为负例0。(注：薪资分两种，分别为>=50K和<=50K，举例：假如用逻辑回归预测，提交结果即为sigmoid函数输出的概率。)

| 字段名 | 含义 |
| - | :-: | 
| Age | 年龄| 
| Workclass | 工作类型| 
| Fnlwgt | 身份背景| 
| Education | 教育程度| 
| education_num | 受教育时间| 
| Maritial_status | 婚姻状况| 
| Occupation | 职业| 
| Relationship | 关系| 
| Race | 种族| 
| Sex | 性别| 
| Capital_gain | 资本收益| 
| Capital_loss | 资本损失| 
| Hours_per_week | 每周工作时间| 
| Native_country | 原籍| 
| income | 收入（<=50K or >=50K）| 
##### 表三 income数据集的字段解释
| 字段名 | 含义 |
| - | :-: | 
| id	| 销售事件唯一编号|
| income_prob	| 薪资>=50K的概率|
##### 表四 需要提交的结果income_result.csv的结构
由提交的结果(income_result.csv)根据对数损失函数(logloss)计算成绩，评分函数为<img src="http://latex.codecogs.com/gif.latex?score=-\frac{1}{N}\sum_{i=1}^{N}(y_{i}log(f_{i})-(1-y_{i})log(1-f_{i}))" title="score=-\frac{1}{N}\sum_{i=1}^{N}(y_{i}log(f_{i})-(1-y_{i})log(1-f_{i}))" />其中<img src="http://latex.codecogs.com/gif.latex?N" title="N" />是预测样本数量， <img src="http://latex.codecogs.com/gif.latex?y_{i}" title="y_{i}" />为样本<img src="http://latex.codecogs.com/gif.latex?i" title="i" />的实际薪资状况(二值变量0或1)，<img src="http://latex.codecogs.com/gif.latex?f_{i}" title="f_{i}" />是你根据训练模型预测样本<img src="http://latex.codecogs.com/gif.latex?i" title="i" />薪资>=50K即为正例的概率，<img src="http://latex.codecogs.com/gif.latex?score" title="score" />越低越好，越低说明预测越准确。[提示:结果不要提交0或1，这会使你的logloss非常非常大，原因见评分公式和评分代码]。提交结果示例如下。

| id | income_prob | 
| - | :-: | 
|1	| 0.55| 
|2	| 0.02| 
|...	| ...|
|N	| 0.99|
##### 表五 需要提交的结果income_result.csv的示例




