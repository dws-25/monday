
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

print( train )  # [10,886행rows x 12columns]  
print()
print( test )   # [6,493 rows x 9 columns]    
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
# train['year'] = train['datetime'].dt.year
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



train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month
train['day'] = train['datetime'].dt.day

# train['datetime'] = train.datetime.apply(pd.to_datetime)
# train['date'] = train.datetime.apply(lambda x: x.date) 
train['date'] = train['datetime'].dt.date
train['dayofweek'] = train['datetime'].dt.day_name() 
# train['dayofweek'] = train['datetime'].dt.day_of_week  # 월요일 0  ~~   
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

#순서3 시간이 엄청 오래걸림 시계열=시간차이  데이터로드 행rows얼마만큼 영향을 주는지확인 차트주식처럼 출력
# print("train데이터 date lineplot test ")
# plt.figure(figsize=(12,4))
# sns.lineplot(data=train, x='date', y='count')
# plt.xticks(rotation=45)
# plt.title("train데이터 date lineplot test ")
# plt.show()  #ok 

print(train) #훈련데이터 시즌
# 순서1] 한화면에  2개 그래프를 표시  시즌별 대여횟수,  요일별 대여횟수  sns.pointplot()
# 훈련train데이터   x=hour  y=대여횟수count, hue='season4계절'
# 훈련train데이터   x=hour  y=대여횟수count, hue='dayofweek'
fig, (ax1,ax2) = plt.subplots(nrows=2)
fig.set_size_inches((16,6))
sns.pointplot(data=train, x='hour', y='count', hue='season',  ax=ax1,  palette='Set1')  #sns.lineplot(data=train, x='date', y='count')
sns.pointplot(data=train, x='hour', y='count', hue='dayofweek', ax=ax2,  palette='Set1')  #sns.lineplot(data=train, x='date', y='count')
plt.show()

# 요일별  대여횟수 , 시간 체크 
# print("train 요일별 대여횟수 pointplot ")
# plt.figure(figsize=(16,6))
# sns.pointplot(data=train, x='hour', y='count', hue='dayofweek',  palette='Set1')  #sns.lineplot(data=train, x='date', y='count')
# plt.show()


#year_month에러발생해서 함수이용해서 컬럼생성  대여횟수   sns.barplot(data=train, x='year_month', y='count' )사용
def year_month(datetime):
    return "{0}-{1}".format(datetime.year, datetime.month)

train['year_month'] = train['datetime'].apply(year_month)

plt.figure(figsize=(16,4))
sns.barplot(data=train,  x='year_month', y='count',  palette='Set1')  #sns.lineplot(data=train, x='date', y='count')
plt.show()

# print( train.info() ) # [10886 rows x 21 columns]
# print()

# 해결boxplot()이용,  lineplot실습, pointplot실습,  barplot실습
# 각자 해결
# fig, axes = plt.subplots(nrows=2,ncols=2) # 2행 * 2열
# fig.set_size_inches(14, 8)
# data=train,  y='count',  x='season/hour/workingday/year_month '  palette='Set1'


# 대여cnt수치를 보면 봄에 수요적고,  가을,여름,겨울순으로...
# 참고 print('점심식사후 boxplot연습 ')
# 참고 sns.boxplot(data=train, y='count', x='season' ,palette='Set1')
# 참고 plt.show()
print( train )          # [10,886  rows  x  20 columns]
# 처음꺼 print( train )  # [10,886 rows  x   12columns]  
print()
print()


def my_removeIQR(df,  col):
    Q1 = np.percentile( df[col], 25)
    Q3 = np.percentile( df[col], 75)
    IQR = Q3 - Q1 
    step = 1.5 * IQR
    ret = df[ (df[col] >= Q1 - step) & (df[col] <= Q3 + step) ]
    return ret 
 
train = my_removeIQR(train, 'count')
print(': ' * 50)
print( train)    #줄어듬[10586 rows x 21 columns]
print(': ' * 50)
print()


#'dayofweek'제외
categorical_feature_names=['season', 'holiday', 'workingday','weather','month','year','hour']
for var in categorical_feature_names:
    train[var] = train[var].astype('category')
    test[var] = test[var].astype('category')

feature_names = [ 'season' , 'holiday', 'workingday','weather','month','year','hour']
X_train = train[feature_names]
X_test  = test[feature_names]
label_name = "count" 
y_train = train[label_name]

from sklearn.ensemble import RandomForestRegressor 
model = RandomForestRegressor(n_estimators=100, random_state=0)
print('학습모델 생성', model)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('예측값 ', y_pred[0:10])  # [ 7.97100971   3.09747092   4.15696312  27.24800577  75.21720169 180.42985782 113.37132937]


#훈련데이터 원래 갯수  [10,886행rows x 12columns]  03-18-화요일아침  [10586 rows x 21 columns]

'''
X_train = train.drop('count', axis=1).astype(np.float32)
y_train = train['count'].astype(np.float32) 
X_test = test.astype(np.float32)
rf_reg = RandomForestRegressor(
    n_estimators=500, # 트리 수          max_depth=15, # 트리 수
    min_samples_split=5, # 트리 수       min_samples_leaf=1, # 최소 리프 노드(제일 끝 노드) 샘플 수
    max_features='sqrt', # 최대 특성 수   bootstrap=True, # 부트스트랩 샘플링 (True면 중복을 허용)
    random_state=42 # 난수 시드 설정 값
)
rf_reg.fit(X_train, y_train)
y_pred = rf_reg.predict(X_test) 결과  0.93


# 이상치 제거
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_filtered
train = remove_outliers_iqr(train, 'count')
'''
#pip install autoviz
#pip install sweetviz

import sweetviz as sv
from autoviz.AutoViz_Class import AutoViz_Class

AutoViz_Class().AutoViz(X_train)
report = sv.analyze(X_train)
report.show_html('./bike/sweetviz_report_df.html')
#자동으로 html문서를 직접 실행

import os
import webbrowser
path = './bike/sweetviz_report_df.html'
webbrowser.open(os.path.realpath(path))

print(path, '파일 오픈 실행까지 확인했습니다')