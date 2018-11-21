# 读取txt文件
import numpy as np
from numpy import *
#逐行处理，存入矩阵
def file2matrix(Mat,lines):
    # length - 1是因为去除第一行的属性名称
    index = 0#标签变量
    for line in lines[1:]:#从第一行以后开始，不包含第一行属性名称
        line = line.strip()
        #Python strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
        # 该方法只能删除开头或是结尾的字符，不能删除中间部分的字符。
        # split line according /t and return a list
        listFromLine = line.split('\t')
        #str.split(str="", num=string.count(str))通过指定分隔符对字符串进行切片，
        # 如果参数 num 有指定值，则仅分隔 num 个子字符串
        Mat[index, :] = listFromLine#按行装进Mat
        index += 1
    return Mat

#训练数据的读取
train = open('zhengqi_train.txt')
# get all lines
lines_train = train.readlines()
length_train = len(lines_train)#计算行数
# length 行， 39列
Mat_train = zeros((length_train - 1, 39))#创造一个0矩阵来装处理过的数据，
datax = file2matrix(Mat_train,lines_train)  # 包含标签的所有数据
y = datax[:, -1]  # 标签
data1 = datax[:, 0:38]  # 除去标签剩下的属性数据

#测试数据的读取(测试数据不含标签)
test = open('zhengqi_test.txt')
# get all lines
lines_test = test.readlines()
length_test = len(lines_test)#计算行数
# length 行， 38列
Mat_test = zeros((length_test - 1, 38))#创造一个0矩阵来装处理过的数据，
testx = file2matrix(Mat_test,lines_test)  # 包含标签的所有数据
#调用函数
from sklearn.linear_model import LinearRegression

linreg = LinearRegression()
#训练
linreg.fit(data1, y)
# 预测
y_pred = linreg.predict(testx)
#输出结果
print(y_pred)
print(type(y_pred))
#保存为1列txt文件
np.savetxt("prediction.txt", y_pred)

'''
#评估
from sklearn.metrics import mean_absolute_error
print('平均绝对误差:',mean_absolute_error(y_test, result))

from sklearn.metrics import mean_squared_error
print('平均方差:',mean_squared_error(y_test, result))

'''



