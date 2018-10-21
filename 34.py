# -*- coding: utf-8 -*-
from __future__ import print_function
import pandas as pd

catering_sale = 'C:/Users/Tang/Desktop/dM/chapter3/demo/data/catering_sale_all.xls'  # 餐饮数据
data = pd.read_excel(catering_sale, index_col=u'日期')  # 读取数据，指定“日期”列为索引列
data.corr()
data.corr()[u'百合酱蒸凤爪']
data[u'百合酱蒸凤爪'].corr(data[u'翡翠蒸香茜饺'])