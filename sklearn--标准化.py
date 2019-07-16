# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 10:07:08 2019

@author: 75286
"""

# 标准化数据模块
from sklearn import preprocessing 
import numpy as np

# 将资料分割成train与test的模块
from sklearn.model_selection import train_test_split

# 生成适合做classification资料的模块
from sklearn.datasets.samples_generator import make_classification 

# Support Vector Machine中的Support Vector Classifier
from sklearn.svm import SVC 

# 可视化数据的模块
import matplotlib.pyplot as plt 

##建立Array
#a = np.array([[10, 2.7, 3.6],
#              [-100, 5, -2],
#              [120, 20, 40]], dtype=np.float64)
#
##将normalized后的a打印出
#print(preprocessing.scale(a))

#生成具有2种属性的300笔数据
X, y = make_classification(
    n_samples=300, n_features=2,
    n_redundant=0, n_informative=2, 
    random_state=22, n_clusters_per_class=1, 
    scale=100)

#归一化后，准确度提升了很多
#X=preprocessing.scale(X)
#指定范围
X=preprocessing.minmax_scale(X,feature_range=(-1,1))


#可视化数据
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()


#训练并测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
clf = SVC()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))