# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 10:32:04 2019

@author: 75286
"""

from sklearn.datasets import load_iris # iris数据集
from sklearn.model_selection import train_test_split # 分割数据模块
from sklearn.neighbors import KNeighborsClassifier # K最近邻(kNN，k-NearestNeighbor)分类算法

#加载iris数据集
iris = load_iris()
X = iris.data
y = iris.target
#建立模型
knn = KNeighborsClassifier()

##part1 普通的训练加测试
##分割数据并
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
##训练模型
#knn.fit(X_train, y_train)
#
##将准确率打印出
#print(knn.score(X_test, y_test))

##part2 交叉验证方式
from sklearn.model_selection import cross_val_score
##使用K折交叉验证模块,cv表示分几组
#scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
##将5次的预测准确率打印出
#print(scores)
##将5次的预测准确平均率打印出
#print(scores.mean())

#part3改变K近邻的neibor值，看准确度的改变
import matplotlib.pyplot as plt
k_range=range(1,31)
k_scores=[]
k_loss=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
#    accuracy表示准确度
    scores=cross_val_score(knn,X,y,cv=10,scoring='accuracy')
    k_scores.append(scores.mean())
#    neg_mean_squared_error表示平均方差，一般用它来判断Regression的好坏
    loss = -cross_val_score(knn,X,y,cv=10,scoring='neg_mean_squared_error')
    k_loss.append(loss.mean())    
plt.subplot(2,1,1)
plt.plot(k_range,k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')

plt.subplot(2,1,2)
plt.plot(k_range,k_loss)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated mean_squared_error')
    












