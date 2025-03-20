
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

# 사이킷럿(scikit-learn) 라이브러리 임포트 
#--------------------------------------------------------------------------------------------
train = pd.read_csv('./bike/train.csv')
test = pd.read_csv('./bike/test.csv')
#parse_dates=['datetime'] 데이터 불러올 때 이렇게 하면 lambda x: x.date가 안먹힐 수 잇음
#따라서 train['datetime'] =train.datetime.apply(pd.to_datetime) 처럼 따로 선언하거나
#그냥 dt.year쓰던가 ㅇㅋ?

train['datetime'] =train.datetime.apply(pd.to_datetime)
train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month
train['day'] = train['datetime'].dt.day
train['date'] = train.datetime.apply(lambda x: x.date) #람다식
train['dayofweek'] = train['datetime'].dt.day_name() 
train['hour'] = train['datetime'].dt.hour
train['minute'] = train['datetime'].dt.minute
train['quarter'] = train['datetime'].dt.quarter
print( train )  
print()  #ok 


print("람다식 적용 date컬럼 11:10  train 요일별 대여횟수 pointplot ")
plt.figure(figsize=(16,6))
sns.pointplot(data=train, x='hour', y='count', hue='dayofweek',  palette='Set1')  #sns.lineplot(data=train, x='date', y='count')
plt.show()



print()
print()