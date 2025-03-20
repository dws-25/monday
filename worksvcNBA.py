# import matplotlib.pyplot as plt        # 첫번째
# from  matplotlib import pyplot as plt  # 두번째
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
from sklearn.datasets import make_classification 
from sklearn import svm # 서포트 벡터 머신
# from sklearn.svm import SVC # 서포트 벡터 머신
from sklearn.tree import DecisionTreeClassifier # 결정트리
from sklearn.linear_model import LogisticRegression # 로지스틱 회귀
from sklearn.ensemble import RandomForestClassifier # 랜덤 포레스트
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
#--------------------------------------------------------------------------------------------------------------------------------------
# 순서1] nba_source.csv
df = pd.read_csv('./test/nba_source.csv')
print(df) #[500행rows x 32columns] 
print()
print(df.info())

print('누락갯수')
print(df.isna().sum())
print()
df['3P'].fillna(df['3P'].mean(), inplace=True)
df['3P%'].fillna(df['3P%'].mean(), inplace=True)
df['2P%'].fillna(df['2P%'].mean(), inplace=True)
df['FT%'].fillna(df['FT%'].mean(), inplace=True)
df['TRB'].fillna(df['TRB'].mean(), inplace=True)
df['BLK'].fillna(df['BLK'].mean(), inplace=True)

# Awards 필드는 삭제 
df.drop('Awards', axis=1, inplace=True)

print(df) #[500행rows x 32columns] 
print()
print(df.info())
# print('누락갯수')
# print(df.isna().sum())

# 순서2] [100 rows x 31 columns] 추출  훈련80건 + 테스트20건 
test_df = df.iloc[ : 100]
print(test_df) # [100행rows x 32columns] 
print()
time.sleep(0.5)

# from sklearn.model_selection import train_test_split
# 참고  X, y = make_classification(n_samples=100, n_features=20, n_informative=15, n_redundant=5, random_state=42)
# 참고  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 쌤껏  X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2 )
train, test = train_test_split( test_df , test_size=0.2, random_state=42 )
print(train)
print('train 데이터출력여부확인')
print()
print(test)
print('test 데이터출력여부확인')
print()

# 62번라인 참고해서 
train_data  = train[['3P', 'BLK', 'TRB']]  
train_label = train[['Pos']]

test_data  = test[['3P', 'BLK', 'TRB']]
test_label = test[['Pos']]

# print(train_data )
print('test_label 포지션')
print(test_label)
print('- ' * 50)
print('- ' * 50)

svc = svm.SVC() 
svc.fit( train_data, train_label )
pred_svc  = svc.predict( test_data)
print()
print('pred_svc')
print(pred_svc)

print('test레이블')
# print(test_label.values) 
print(test_label.values.ravel())  #(n,1) =>  (n,)
print()

accuracy = accuracy_score(test_label, pred_svc)
print('예측정확도 비율 ', accuracy) #예측정확도 비율  0.3
print()

report = classification_report(test_label, pred_svc , labels=np.unique(pred_svc))  #, labels=np.unique(pred_svc)
print(report)

#실제데이터레이블/예측데이터레이블 비교  accuracy_score평가  처리 
# 해결 데이터구조, 크기를 보고  방해되는 행이나 컬럼 재정비해서 사용 
# 해결 결측값 갯수 출력, info()정보 
# 해결 결측값은 숫자라서 평균값으로 대체,  Awards필드삭제 
# 해결 [500 rows x 32 columns]대신  [100 rows x 31 columns] 추출후   훈련80건 + 테스트20건
# 해결 다른방식으로 접근 훈련,테스트 데이터 분리 8/2  random_state = 42 
# 정답지 target선정 
# SVC모델생성( C, gamma )  모델생성 괄호안의 매개인자
# fit(훈련x,훈련y), predict(테스트), 실제데이터레이블/예측데이터레이블 비교  accuracy_score평가 


'''
1 C (Regularization Parameter, 정규화 매개변수)
C는 오차 허용 정도를 조절하는 값
C=1: 기본적인 설정으로 적절한 균형을 유지
C=0.1: 더 넓은 마진을 허용 (과소적합 위험 증가)
C=10: 마진을 줄이고 데이터에 더욱 적합하게 학습 (과적합 위험 증가)

2 gamma (Kernel Coefficient, 커널 계수) auto 
'''


print()
print()