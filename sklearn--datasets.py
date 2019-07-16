# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 16:31:10 2019

@author: 75286
"""

from __future__ import print_function
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

loaded_data=datasets.load_boston()
data_X=loaded_data.data
data_y=loaded_data.target

model=LinearRegression()
model.fit(data_X,data_y)

print(model.predict(data_X[:4,:]))#预测的值
print(data_y[:4])#实际的值


#创建虚拟数据－可视化 
X,y=datasets.make_regression(n_samples=100,n_features=1,n_targets=1,noise=10)
plt.scatter(X,y)
