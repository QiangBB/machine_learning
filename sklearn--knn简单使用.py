# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 14:27:30 2019

@author: 75286
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


iris = datasets.load_iris()
iris_X=iris.data  #存属性
iris_y=iris.target  #存标签

#产生训练集、测试集，其中测试集（X_test+y_test）占30%
X_train,X_test,y_train,y_test=train_test_split(iris_X,iris_y,
                                               test_size=0.3)

#建立模型，开始预测
knn=KNeighborsClassifier()
knn.fit(X_train,y_train)

#输出结果对比差异
result=knn.predict(X_test)
print(result)
print(y_test)

one=0
count=0
while one<result.size:
    if result[one]!=y_test[one]:
        count=count+1
    one=one+1

print('正确率为'+str((1-count*1.0/result.size)*100)+'%')