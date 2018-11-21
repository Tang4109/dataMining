# 读取txt文件
import numpy as np
import pandas as pd
from sklearn.decomposition import pca

#训练数据
datax=pd.read_table("zhengqi_train.txt")
#y = np.array(datax)[:, -1]  # 将标签保存在y中
y=datax.values[:, -1]#values自动转为数组,将标签保存在y中
#data1 = datax.drop(['target'], axis=1)  # 将分类标签项丢弃,axis=1表示对列操作
#data2=np.array(data1)  # 转为数组类型
data1=datax.values[:, 0:-1]

#测试数据
testx=pd.read_table("zhengqi_test.txt")
#test1=np.array(testx)  # 转为数组类型
test1=testx.values

# PCA数据处理
pca = pca.PCA(n_components=20)#主元14
pca.fit(data1)
data_pca = pca.transform(data1)
test_pca = pca.transform(test1)



'''
# 分离出训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(data_pca, y, test_size=0.2)#二八分

'''





'''
#调用函数，线性回归训练
from sklearn.linear_model import LinearRegression

linreg = LinearRegression()
#训练
linreg.fit(X_train, Y_train)
# 预测
y_pred = linreg.predict(X_test)


'''
#岭回归训练
from sklearn.linear_model import Ridge
lg = Ridge()
# lg = LinearRegression()
lg.fit(data_pca, y)
y_pred = lg.predict(test_pca)






#输出结果
print(y_pred)
print(type(y_pred))
#保存为1列txt文件
np.savetxt("prediction1.txt", y_pred)
'''
#评估
from sklearn.metrics import mean_absolute_error
print('平均绝对误差:',mean_absolute_error(Y_test, y_pred))
from sklearn.metrics import mean_squared_error
print('平均方差:',mean_squared_error(Y_test, y_pred))

'''

