

# RBF就是高斯核函数


# 高斯过程回归，首先要判断，所求的是否满足正太分布，如果满足，就可以用高斯正太回归。可以参考一下代码


import numpy as np

import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C  # REF就是高斯核函数


np.set_printoptions(threshold=np.inf)
# 创建数据集
#testx=np.loadtxt('C:/Users/Tang/Desktop/高斯回归/pf/WFG3P4.3D.pf')
#test1=testx[1:1999]


datax= np.loadtxt('C:/Users/Tang/Desktop/高斯回归/pf/WFG1P3.3D.pf')
data1=datax[1:1952]
from sklearn.model_selection import train_test_split
#dat为数据集,含有feature和label.
train, test = train_test_split(data1, test_size = 0.2)
data = train
y=test[:, -1]
test1 = test[:,:-1]

# 核函数的取值

kernel = C(0.8, (0.001, 1)) * RBF(0.5, (1e-4, 20))
# 创建高斯过程回归,并训练

reg = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=1e-12)#alpha代表噪声
reg.fit(data[:, :-1], data[:, -1])  # 这是拟合高斯过程回归的步骤，data[:,:-1]获取前两列元素值，data[:,-1]获取最后一列元素的值

# 创建一个作图用的网格的测试数据，数据位线性，x为【1982，2009】间隔位0.5；y为【57.5，165】间隔位0.5

x_min, x_max = data[:, 0].min(), data[:, 0].max()  # 获取data的第一列的最小值减1和最大值加1

y_min, y_max = data[:, 1].min(), data[:, 1].max()  # 获取data的第二列的最小值减1和最大值加1

xset, yset = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))


# 作图，并画出

fig = plt.figure(figsize=(10.5, 5))
from mpl_toolkits.mplot3d import Axes3D  # 实现数据可视化3D
ax1 = fig.add_subplot(111,projection='3d')  # 画上一个1*2的图形，在第一个位置，这就是121的含义


result=reg.predict(test1)
ax1.scatter(test[:, 0], test[:, 1], y,marker = 'o',s=2,c='r')  # scatter表示画分散的点
ax1.scatter(test[:, 0], test[:, 1], result,marker = 'o',s=10, c='g')  # scatter表示画分散的点



print(result)

from sklearn.metrics import mean_absolute_error
print('平均绝对误差:',mean_absolute_error(y, result))

from sklearn.metrics import mean_squared_error
print('平均方差:',mean_squared_error(y, result))



ax1.set_title('y')#第一个值

ax1.set_xlabel('x1')

ax1.set_ylabel('x2')

plt.show()





