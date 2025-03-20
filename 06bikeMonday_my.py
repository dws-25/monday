
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
train = pd.read_csv('./bike/train.csv' ,  parse_dates=['datetime'] )
test = pd.read_csv('./bike/test.csv'  ,   parse_dates=['datetime'] )

print( train )  # [10,886행rows x 12columns]  #ROC성능 테스트 최적 
print()
print( test )   # [6,493 rows x 9 columns]    #ROC성능 테스트 최적 
print()


arr = np.array([1, 2, 4, 10, 100, 1000])  
log_arr = np.log(arr) 
# 1,     2,         4,            10숫자       100숫자       1000숫자
# [0.    0.69314718  1.38629436   2.30258509   4.60517019   6.90775528 ]
print("log(arr) =", log_arr)            
print()
print()


#첨도 skew() 음수분포가 오른쪽으로, 양수분포가 왼쪽으로 치우침
#왜도 kurt() 0이면 정규분포, 0보다크면 정규분포보다 뾰족한분포
# plt.figure(figsize=(12,7))
# fig, ax = plt.subplots(1,1, figsize = (12, 6))  
# sns.distplot(train['count'], color='hotpink',  label=train['count'].skew(),ax=ax)
# plt.title(" train['count'].skew()")
# plt.legend()
# plt.show()
# print('전 왜도 kurt() 측정' , train['count'].kurt()) # 1.3000929518398334
# print('전 첨도 skew() 측정' , train['count'].skew()) # 1.2420662117180776
# print()

# # plt.figure(figsize=(12,7))
# fig, ax = plt.subplots(1,1, figsize = (12, 6))  
# train['count_log'] = train['count'].map(lambda i:np.log(i) if i > 0 else 0)
# sns.distplot(train['count_log'], color='green',  label=train['count_log'].skew(),ax=ax)
# plt.title("후후 train['count_log'].skew() ")
# plt.legend()
# plt.show()

# print('후 왜도 kurt() 측정' , train['count_log'].kurt()) # 0.24662183416964112
# print('후 첨도 skew() 측정' , train['count_log'].skew()) # -0.9712277227866112
# print()


# train정답 
# train['year'] = train['datetime'].apply(lambda x : x.year)
# train['month'] = train['datetime'].dt.month
# train['day'] = train['datetime'].dt.day
# train['date'] = train['datetime'].dt.date
# # train['dayofweek'] = train['datetime'].dt.day_name()    
# train['dayofweek'] = train['datetime'].dt.day_of_week  
# train['hour'] = train['datetime'].dt.hour
# train['minute'] = train['datetime'].dt.minute
# train['quarter'] = train['datetime'].dt.quarter
# print( train )  
# print()  #ok 


# train람다식, apply함수 적용
# train['datetime'] = train.datetime.apply(pd.to_datetime)
# train['date'] = train.datetime.apply(lambda x: x.date) #.dt.date
    
train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month
train['day'] = train['datetime'].dt.day
train['date'] = train['datetime'].apply(lambda dt: dt.date)
# train['dayofweek'] = train['datetime'].dt.day_name()    
train['dayofweek'] = train['datetime'].dt.day_of_week  
train['hour'] = train['datetime'].dt.hour
train['minute'] = train['datetime'].dt.minute
train['quarter'] = train['datetime'].dt.quarter
print( train )  
print()  #ok 



#test데이터 항목 그대로 유지 
test['year'] = test['datetime'].dt.year
test['month'] = test['datetime'].dt.month
test['day'] = test['datetime'].dt.day
test['date'] = test['datetime'].dt.date
# test['dayofweek'] = test['datetime'].dt.day_name()    
test['dayofweek'] = test['datetime'].dt.day_of_week  
test['hour'] = test['datetime'].dt.hour
test['minute'] = test['datetime'].dt.minute
test['quarter'] = test['datetime'].dt.quarter
print( test )  
print()  #ok 


print( train )  # [10,886행rows x 20columns] 
print()
print( test )   # [6,493 rows x 17 columns]    
print(' -' * 40)
print()

print( train.info() )

# fig, (ax1,ax2) = plt.subplots(nrows=2)
# fig.set_size_inches((16,6))
# sns.pointplot(data=train, x='hour', y='count', hue='season',  ax=ax1,  palette='Set1')  #sns.lineplot(data=train, x='date', y='count')
# sns.pointplot(data=train, x='hour', y='count', hue='dayofweek', ax=ax2,  palette='Set1')  #sns.lineplot(data=train, x='date', y='count')
# plt.show()

'''
plt.figure(figsize=(16,6))
sns.pointplot(data=train, x='hour', y='count', hue='dayofweek',  palette='Set1')  #sns.lineplot(data=train, x='date', y='count')
plt.show()
'''

'''
plt.figure(figsize=(12,4))
sns.pointplot(data=train, x='hour', y='count', hue='season', palette='Set1')  #sns.lineplot(data=train, x='date', y='count')
plt.show()
plt.figure(figsize=(12,4))
sns.pointplot(data=train, x='hour', y='count', hue='dayofweek', palette='Set2')  #sns.lineplot(data=train, x='date', y='count')
plt.show()
'''

#gpt랑 내가 한거 왜 에러일까
# plt.figure(figsize=(12,5))
# plt.subplot(2,1,1)
# sns.pointplot(date=train, x ='hour', y='count', hue='season', errorbar=None)
# plt.title("시즌별 대여 횟수")

# plt.subplot(2, 1, 2)  # 1행 2열 중 두 번째
# sns.pointplot(data=train, x='hour', y='count', hue='dayofweek', errorbar=None)
# plt.title("요일별 대여 횟수")

# plt.tight_layout()  # 레이아웃 자동 조정
# plt.show()


#순서3 시간이 엄청 오래걸림 시계열=시간차이  데이터로드 행rows얼마만큼 영향을 주는지확인 차트주식처럼 출력
# plt.figure(figsize=(16,4))
# sns.lineplot(data=train, x='date', y='count')
# plt.xticks(rotation=45)
# plt.title("train데이터 date lineplot test ")
# plt.show()  #ok 

# 순서1] 한화면에 2개 그래프를 표시 시즌별 대여횟수, 요일별 대여횟수 sns.pointplot()
# 훈련train데이터 x=hour y=대여횟수count, hue = 'season4계절'
# 훈련train데이터 x=hour y=대여횟수count, hue = 'dayofweek'

#식사 후 문제 lineplot

# sns.countplot(x='day', data=train, palette='Set1')
# plt.show()

# sns.barplot(x='sex', y='tip', data= train, palette='Set2')
# plt.show()


# sns.boxplot(x='time', y='total_bill', data=train, palette='Set3')
# plt.show()

# sns.histplot( x='total_bill', data=train, palette='Set1')
# plt.show()

# sns.scatterplot( x='total_bill', y='tip', data=train,  c='green')
# plt.show()

#문제 lineplot, pointplot, barlot,boxplot

# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16,10)) #2행 *2열
# # fig.set_size_inches(14,8)
# sns.scatterplot(data=train, x='month', y='count' ,hue ='dayofweek', palette = 'Set1',ax= axes[0,0])
# axes[0,0].set_title('월간 대여 수')

# sns.pointplot(data=train, x='season', y='count',hue='month' ,palette = 'Set2',ax= axes[0,1])
# axes[0,1].set_title('Your Title Here')

# sns.boxplot(data=train, x='holiday', y='count',palette = 'Set3',ax= axes[1,0])
# axes[1,0].set_title('Your Title Here')

# sns.lineplot(data=train, x='workingday', y='count',hue='dayofweek' ,palette = 'Set3',ax= axes[1,1])
# axes[1,1].set_title('Your Title Here')

# plt.tight_layout
# plt.show()


# 첫번째  학습 모델 선택 SVC,LR, LR,
# 두번째  train데이터 test데이터 분리  X = [온도,습도,강풍,요일,시즌 ] ,y = count대여횟수
# 세번째  각자응용 이상치 count 1~395
categorical_feature_names = [ 'season' , 'holiday', 'workingday','weather','dayofweek','month','year','hour' ]
for var in categorical_feature_names:
    train[var] = train[var].astype('category')
    test[var] = test[var].astype('category')
print( train.info() )  
print() 
feature_names = [ 'season' , 'holiday', 'workingday','weather','dayofweek','month','year','hour' ]

X_train = train[feature_names]
X_test = test[feature_names]
label_name = "count"
y_train = train[label_name]
print(y_train )

from sklearn.ensemble import RandomForestRegressor  #영임쌤 새로운 학습모델 
model = RandomForestRegressor(n_estimators=100, random_state= 42)
print(' 학습모델 생성 ', model)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(' 예측값 ',y_pred )
print()

#이상치 제거 count필드
from collections import Counter
def my_dropIQR(df, n, feature):
    outer = []
    for col in feature:
        Q1 = 7
        Q3 = 7
        IQR = Q3 - Q1
        step = 1.5 * IQR
        outer_col = df[ (df[col] < Q1 - step) | (df[col] > Q1 + step) ] .index



print()
print()