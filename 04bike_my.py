import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib import rc

# 음수표기 관리
import matplotlib as mpl
mpl.rc('axes', unicode_minus=False)
mpl.rcParams['axes.unicode_minus']=False

font_name = font_manager.FontProperties(fname='c:/windows/Fonts/malgun.ttf').get_name()
rc('font', family=font_name)

import pandas as pd
import numpy as np
import seaborn as sns 
import time
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# print(train)
# print()
# print(test)
# print(train.info()) #결측없다. 전처리-> datetimetime필드 컬럼, count 숫자 완화 np.log() 사용
# print()
# print(test.info())
# print()

train = pd.read_csv('bike/train.csv', parse_dates=['datetime'])
test = pd.read_csv('bike/test.csv', parse_dates=['datetime'])

'''
                  datetimetime  season  holiday  workingday  weather  ...  humidity  windspeed  casual  registered  count
0      2011-01-01 00:00:00       1        0           0        1  ...        81     0.0000       3          13     16
1      2011-01-01 01:00:00       1        0           0        1  ...        80     0.0000       8          32     40
2      2011-01-01 02:00:00       1        0           0        1  ...        80     0.0000       5          27     32
3      2011-01-01 03:00:00       1        0           0        1  ...        75     0.0000       3          10     13
4      2011-01-01 04:00:00       1        0           0        1  ...        75     0.0000       0           1      1
...                    ...     ...      ...         ...      ...  ...       ...        ...     ...         ...    ...
10881  2012-12-19 19:00:00       4        0           1        1  ...        50    26.0027       7         329    336
10882  2012-12-19 20:00:00       4        0           1        1  ...        57    15.0013      10         231    241
10883  2012-12-19 21:00:00       4        0           1        1  ...        61    15.0013       4         164    168
10884  2012-12-19 22:00:00       4        0           1        1  ...        61     6.0032      12         117    129
10885  2012-12-19 23:00:00       4        0           1        1  ...        66     8.9981       4          84     88

'''
# train['datetime'] = pd.to_datetime(train['datetime'])
# test['datetime'] = pd.to_datetime(test['datetime'])
# print(train.info())

train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month
train['date'] = train['datetime'].dt.date
train['dayofweek'] = train['datetime'].dt.day_name( )# train['dayofweek'] = train['datetime'].dt.day_of_week
train['hour'] = train['datetime'].dt.hour
train['minute'] = train['datetime'].dt.minute
train['quarter'] = train['datetime'].dt.quarter

X = train.drop('count',axis =1)
y = train['count']



#해결 TRAIN훈련데이터 COUNT열 값 데이터 1~336조절 그냥사용, 이상치, scaler, np.log

#시계열=시간차이
# plt.figure(figsize=(20,6))
# sns.lineplot(data=train, x='date', y='count')
# plt.xticks(rotation=45)
# plt.show()

#distplot()이용해서 skew()첨도를 그래프표시
plt.figure(figsize=(12,8))
print('왜도 skew()측정', train['count'].skew())
print('첨도 kurt()측정', train['count'].kurt())
print()
sns.displot(train['count'], color= 'darkorchid', label=train['count'].skew())
plt.title('왜도 skew 쏠림 =치우침')
# sns.distplot(train['count'].kurt())
plt.legend()
plt.show()
# train['count'].skew()
# train['count'].kurt()

# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state = 48)



print()
print()