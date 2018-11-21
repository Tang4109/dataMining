#测试数据执行与训练数据相同的前期处理

import pandas as pd  # 数据分析

# 读入训练数据，sheetname=1表示第二个表单
test_train = pd.read_excel("C:/Users/Tang/Desktop/模式识别作业/homework2018/ML_data2.xlsx", sheetname=1)

#将国籍为美国的标为1，非美国标为0
test_train.native_country.loc[test_train.native_country == 'United-States']= 1
test_train.native_country.loc[test_train.native_country != 1]= 0

#将性别为男性的标为1，女性标为0
test_train.sex.loc[test_train.sex == 'Male']= 1
test_train.sex.loc[test_train.sex != 1]= 0


# 因为逻辑回归建模时，需要输入的特征都是数值型特征
# 我们先对类目型的特征离散/因子化，采用one-hot编码
# 平展属性
# 我们使用pandas的get_dummies来完成这个工作，并拼接在原来的data_train之上，如下所示
dummies_workClass = pd.get_dummies(test_train['workClass'], prefix='workClass')
dummies_education = pd.get_dummies(test_train['education'], prefix='education')
dummies_marital_status = pd.get_dummies(test_train['marital_status'], prefix='marital_status')
dummies_occupation = pd.get_dummies(test_train['occupation'], prefix='occupation')
dummies_relationship = pd.get_dummies(test_train['relationship'], prefix='relationship')
dummies_race = pd.get_dummies(test_train['race'], prefix='race')

#拼接
df = pd.concat([test_train, dummies_workClass, dummies_education, dummies_marital_status, dummies_occupation,
                dummies_relationship, dummies_race], axis=1)
#丢弃
df.drop(['workClass', 'education', 'marital_status', 'occupation', 'relationship', 'race'], axis=1, inplace=True)
#归一化，为了加快梯度下降时的迭代速度
df.age=(df.age-df.age.min())/(df.age.max()-df.age.min())
df.fnlwgt=(df.fnlwgt-df.fnlwgt.min())/(df.fnlwgt.max()-df.fnlwgt.min())
df.education_num=(df.education_num-df.education_num.min())/(df.education_num.max()-df.education_num.min())
df.capital_gain=(df.capital_gain-df.capital_gain.min())/(df.capital_gain.max()-df.capital_gain.min())
df.capital_loss=(df.capital_loss-df.capital_loss.min())/(df.capital_loss.max()-df.capital_loss.min())
df.hours_per_week=(df.hours_per_week-df.hours_per_week.min())/(df.hours_per_week.max()-df.hours_per_week.min())

#保存新数据
df.to_csv("C:/Users/Tang/Desktop/模式识别作业/homework2018/data_test.csv", index=False)