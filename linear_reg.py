import numpy as np
import pandas as pd
#读取txt文件
from numpy import *

def file2matrix(filename):
    fr = open(filename)
    #get all lines
    lines = fr.readlines()
    length = len(lines)
    #length rows and 38 columns
    returnMat = zeros((length-1, 39))
    index = 0
    for line in lines[1:]:
        #get rid of spaces etc.
        line = line.strip()
        #split line according /t and return a list
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine
        index += 1
    return returnMat

datax=file2matrix('zhengqi_train.txt')#包含标签的所有数据
y=datax[:,-1]#标签
data1=datax[:,0:38]#除去标签剩下的属性数据
#print(datax)

#将数据随机分为训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data1, y, random_state=1)
#默认0.75和0.25
from sklearn.linear_model import LinearRegression

linreg = LinearRegression()

linreg.fit(X_train, y_train)
#print(linreg.intercept_)
#print(linreg.coef_)
#预测
y_pred = linreg.predict(X_test)
print(y_pred)

#评价
from sklearn.metrics import mean_absolute_error
print('平均绝对误差:',mean_absolute_error(y_test, y_pred))

from sklearn.metrics import mean_squared_error
print('平均方差:',mean_squared_error(y_test, y_pred))


