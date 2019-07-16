# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 14:51:38 2019

@author: 75286
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation

#正弦函数的显示
#x=np.arange(0,np.pi*3,0.1)
#y=np.sin(x)
#
#plt.plot(x,y,'ob').show()



#subplot的使用
#x=np.arange(0,np.pi*3,0.1)
#y_sin=np.sin(x)
#y_cos=np.cos(x)
#
#plt.subplot(2,1,1) #高为2，宽为1，第1个
#plt.title('this is sin func')
#plt.plot(x,y_sin)
#
#plt.subplot(2,1,2) #高为2，宽为1，第2个
#plt.title('this is cos func')
#plt.plot(x,y_cos)
#plt.show()



##bar 的使用
#x1=[1,2,3]
#y1=[7,8,9]
#x2=[5,6,7]
#y2=[6,7,8]
#plt.bar(x1,y1,color='r')
#plt.bar(x2,y2,color='b')



##histogram数据的频率分布的图形表示
#a=np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27])
#hist,bins=np.histogram(a,[0,20,40,60,80,100])
#print(hist,bins)# hist 保存各个频率，bins保存各个频率段
#plt.hist(a,bins)# hist函数直接将数据和频率段处理


##传入的参数是数组
#x=np.zeros([2])
#y=np.zeros([2])
#y[0]=10
#y[1]=20
#x[0]=0
#x[1]=0
#plt.plot(x,y,'r',lw=10) # lw调整宽度


#x = np.linspace(-1, 1,66) #切割66份
## 绘制y=2x+1函数的图像
#y = 2 * x + 1
#plt.plot(x, y)
#plt.show()
#
## 绘制x^2函数的图像
#y = x**2
#plt.plot(x, y)
#plt.show()



#x=np.linspace(-1,1,50)
#y1=x*2+1
#plt.figure()
#plt.plot(x,y1)
#plt.figure()# 相当于新建了一个画布
#y2=x**2
#plt.plot(x,y2)


##画在一张画布上
#plt.figure(num=4,figsize=(4,4))
##设置xy坐标的限制
#plt.xlim((-1,1))
#plt.ylim((0,3))
##中文输出
#plt.xlabel(u'这是x轴',fontproperties='SimHei',fontsize=14)
#plt.ylabel(u'这是y轴',fontproperties='SimHei',fontsize=14)
#
##获取当前的坐标框,gca means get current axis
#ax=plt.gca()
#ax.spines['right'].set_color('none')
#ax.spines['top'].set_color('none')
##设置x为下边框
#ax.xaxis.set_ticks_position('bottom')
##设置x为左边框
#ax.yaxis.set_ticks_position('left')
## 设置x轴, y周在(0, 0)的位置
#ax.spines['bottom'].set_position(('data',0))
#ax.spines['left'].set_position(('data',0))
#plt.plot(x,y1)
#plt.plot(x,y2,'r',linewidth='1.0',linestyle = '--')
##划分刻度
#plt.xticks(np.linspace(-1,1,7)) 
#plt.yticks(np.linspace(0,3,7))
## 设置坐标轴label的大小，背景色等信息
#for label in ax.get_xticklabels() + ax.get_yticklabels():
#    label.set_fontsize(12)
#    label.set_bbox(dict(facecolor = 'green', edgecolor = 'None', alpha = 0.7))


#x0=1
#y0=2*x0+1
##画一个坐标点
#plt.scatter(x0,y0,s=50,color='r')
##绘制虚线
#plt.plot([x0,y0],[y0,0],'k--',lw=2.0)
## 绘制注解一
#plt.annotate(r'$2 * x + 1 = %s$' % y0, xy = (x0, y0), xycoords = 'data', xytext = (+30, -30), \
#             textcoords = 'offset points', fontsize = 16, arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3, rad = .2'))
#


##绘制散点图
## 数据个数
#n = 1024
## 均值为0, 方差为1的随机数
#x = np.random.normal(0, 1, n)
#y = np.random.normal(0, 1, n)
#
## 计算颜色值
#color = np.arctan2(y, x)
## 绘制散点图
#plt.scatter(x, y, s = 75, c = color, alpha = 0.5)
## 设置坐标轴范围
#plt.xlim((-1.5, 1.5))
#plt.ylim((-1.5, 1.5))
#
## 不显示坐标轴的值
#plt.xticks(())
#plt.yticks(())


## 绘制等高线图
## 定义等高线高度函数
#def f(x, y):
#    return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(- x ** 2 - y ** 2)
#
## 数据数目
#n = 256
## 定义x, y
#x = np.linspace(-3, 3, n)
#y = np.linspace(-3, 3, n)
#
## 生成网格数据
#X, Y = np.meshgrid(x, y)
#
#
## 填充等高线的颜色, 8是等高线分为几部分
#plt.contourf(X, Y, f(X, Y), 8, alpha = 0.75, cmap = plt.cm.hot)
## 绘制等高线
#C = plt.contour(X, Y, f(X, Y), 8, colors = 'black', linewidth = 0.5)
## 绘制等高线数据
#plt.clabel(C, inline = True, fontsize = 10)
#
## 去除坐标轴
#plt.xticks(())
#plt.yticks(())
#plt.show()


## 定义图像数据
#a = np.linspace(0, 1, 9).reshape(3, 3)
## 显示图像数据
#plt.imshow(a, interpolation = 'nearest', cmap = 'bone', origin = 'lower')
## 添加颜色条
#plt.colorbar()
## 去掉坐标轴
#plt.xticks(())
#plt.yticks(())
#plt.show()



#from mpl_toolkits.mplot3d import Axes3D
## 定义figure
#fig = plt.figure()
## 将figure变为3d
#ax = Axes3D(fig)
#
## 数据数目
#n = 256
## 定义x, y
#x = np.arange(-4, 4, 0.25)
#y = np.arange(-4, 4, 0.25)
## 生成网格数据
#X, Y = np.meshgrid(x, y)
#
## 计算每个点对的长度
#R = np.sqrt(X ** 2 + Y ** 2)
## 计算Z轴的高度
#Z = np.sin(R)
#
## 绘制3D曲面
#ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, cmap = plt.get_cmap('rainbow'))
## 绘制从3D曲面到底部的投影
#ax.contour(X, Y, Z, zdim = 'z', offset = -2, cmap = 'rainbow')
#
## 设置z轴的维度
#ax.set_zlim(-2, 2)
#
#plt.show()



##动画的演示
## 定义figure
#fig, ax = plt.subplots()
#
## 定义数据
#x = np.arange(0, 2 * np.pi, 0.01)
## line, 表示只取返回值中的第一个元素
#line, = ax.plot(x, np.sin(x))
#
## 定义动画的更新
#def update(i):
#    line.set_ydata(np.sin(x + i/10))
#    return line,
#
## 定义动画的初始值
#def init():
#    line.set_ydata(np.sin(x))
#    return line,
#
## 创建动画
#ani = animation.FuncAnimation(fig = fig, func = update, init_func = init, interval = 10, blit = False, frames = 200)
#
## 展示动画
#plt.show()





