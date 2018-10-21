# -*- coding: utf-8 -*-
from __future__ import print_function
import pandas as pd

dish_profit = 'C:/Users/Tang/Desktop/dM/chapter3/demo/data/catering_dish_profit.xls'  # 餐饮数据
data = pd.read_excel(dish_profit, index_col=u'菜品名')  # 读取数据，指定“菜品名”列为索引列
data = data[u'盈利'].copy()
data.sort_values(ascending=False)  # 下降排序

import matplotlib.pyplot as plt  # 导入图像库

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

plt.figure()
data.plot(kind='bar')
plt.ylabel(u'盈利(元)')
p = 1.0 * data.cumsum() / data.sum()
p.plot(color='r', secondary_y=True, style='-o', linewidth=0)
plt.annotate(format(p[6], '.4%'), xy=(6, p[6]), xytext=(6 * 0.9, p[6] * 0.9),
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))#添加注释
plt.ylabel(u'盈利(比例)')
plt.show()
#总结：贡献度分析图，我对于程序画图还不是太熟悉