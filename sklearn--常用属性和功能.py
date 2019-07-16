# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 16:55:37 2019

@author: 75286
"""

from sklearn import datasets
from sklearn.linear_model import LinearRegression

loaded_data=datasets.load_boston()
data_X=loaded_data.data
data_y=loaded_data.target

model=LinearRegression()
model.fit(data_X,data_y)
result=model.predict(data_X[:4,:])
#模型的斜率和截距
print(model.coef_,model.intercept_)
print(model.get_params())
#输出精确度
print(model.score(data_X,data_y))