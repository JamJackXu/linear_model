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
</br>
| 字段名	| 含义 |
|id	|销售事件唯一编号|
|date	|销售日期|
|price|	销售价格|
|Bedrooms	|房间数量|
|Bathrooms|	卫生间数量(0.5表示有一个半功能的卫生间)|
|sqft_living|	起居室内部面积|
|sqft_lot|	占地面积|
|Floor	|房屋含有的层数|
|Waterfront|	是否可以看到水面（能看到江/河/湖/海）|
|View	|房屋整体视觉效果评分(0-4)|
|Condition	|房屋的整体条件评分(1-5)|
|Grade|	房屋结构以及设计评分(1-13)|
|Sqft_above	|地面以上的居住面积|
|sqft_basement|	地下室面积|
|yr_build|	房屋建造时间|
|yr_renovated|	房屋最近一次翻新的时间|
|Zipcode|	房子所在的邮编|
|Lat	|地处位置的纬度|
|Long	|地处位置的经度|
|sqft_living15|	距离房屋最近的15个房子的起居室面积|
|sqft_lot15	|距离房屋最近的15个房子的占地面积|
</br>


