# 逻辑回归分类器程序

import numpy as np
import pandas as pd


# 定义sigmoid函数
def sigmoid(z):
    gz = (1 / (1 + np.exp(-z)))
    return gz


# 梯度下降法求theta
def gradientDescent(x, y, theta, alpha, m, numIterations):  # theta要求的参数，alpha学习率，m样本数，numIterations迭代次数
    xTran = np.transpose(x)  # 转置，便于计算
    h = sigmoid(xTran.dot(theta))
    for i in range(numIterations):
        hypothesis = np.dot(xTran, theta)  # x与theta的内积
        loss = hypothesis - y  # 偏差
        cost = -1 * (1 / m) * (np.log(h).T.dot(y) + np.log(1 - h).T.dot(1 - y)) + (lamb / (2 * m)) * np.sum(
            np.square(theta[1:]))  # 带正则化项的代价函数
        gradient = np.dot(x, loss) / m  # 梯度
        theta[0] = theta[0] - alpha * gradient[0]  # 更新theta 0
        theta[1:] = theta[1:] - alpha * (gradient[1:] + (lamb / m) * theta[1:])  # 更新theta1及之后的theta
        print("Iteration %d | cost :%f" % (i, cost))  # 输出代价值
    return theta  # 返回theta


def predict(x):  # 预测函数
    result = np.dot(theta.T, x) + b
    result = 1 / (1 + np.exp(-result))
    for j in result:
        if j > 0.61:  # 当result>0.61时判断为1，否则为0
            prediction.append(1)
        else:
            prediction.append(0)

    return prediction  # 返回预测结果


# 训练
prediction = []  # 将预测值装入prediction数组
lamb = 1000  # 正则项常数
b = 0  # 截距
df_train = pd.read_csv('data_train.csv')  # 读入训练数据
y = np.array(df_train)[:, 8]  # 将分类标签保存在y中
X = df_train.drop(['salary'], axis=1)  # 将分类标签项丢弃
X = np.array(X)  # 转为数组类型
numIterations = 1000  # 迭代1000次
m, n = X.shape  # 数据大小
theta = np.ones(n)  # 初始theta全为1
alpha = 0.1  # 设置更新速率
X = np.transpose(X)  # 转置，便于计算
theta = gradientDescent(X, y, theta, alpha, m, numIterations)  # 训练得到theta
# 预测
df_test = pd.read_csv('data_test.csv')  # 读入测试数据
y_test = np.array(df_test)[:, 8]  # 将分类标签保存在y_test中
X_test = df_test.drop(['salary'], axis=1)  # 将分类标签项丢弃
X_test = np.transpose(X_test)  # 转置，便于计算
predict_result = predict(X_test)  # 预测

# 保存预测结果
dataframe = pd.DataFrame({'salary': predict_result})
# 将DataFrame存储为csv,index表示是否显示行名，default=True
dataframe.to_csv("C:/Users/Tang/Desktop/模式识别作业/homework2018/predict_result0.csv", index=False)

# 计算准确率
from sklearn.metrics import accuracy_score

print('准确率:', accuracy_score(y_test, predict_result))
print('预测正确的数目', accuracy_score(y_test, predict_result, normalize=False))
