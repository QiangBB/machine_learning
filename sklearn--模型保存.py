# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 14:54:01 2019

@author: 75286
"""

from sklearn import svm
from sklearn import datasets

clf=svm.SVC()
iris=datasets.load_iris()
X,y=iris.data,iris.target
clf.fit(X,y)

#保存模型的2种方式

##method one：pickle
#import pickle
#
##保存Model(注:save文件夹要预先建立，否则会报错)
#with open('save_model/clf.pickle','wb') as f:
#    pickle.dump(clf,f)
##读取Model
#with open('save_model/clf.pickle', 'rb') as f:
#    clf2 = pickle.load(f)
#    #测试读取后的Model
#    print(clf2.predict(X[0:1]))


#method two:使用 joblib 保存
from sklearn.externals import joblib #jbolib模块
#保存Model(注:save文件夹要预先建立，否则会报错)
joblib.dump(clf, 'save_model/clf.pkl')

#读取Model
clf3 = joblib.load('save_model/clf.pkl')

#测试读取后的Model
print(clf3.predict(X[0:1]))