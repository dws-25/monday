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
import time

# 사이킷럿(scikit-learn) 라이브러리 임포트 
from sklearn.datasets import make_classification 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC     #from sklearn import svm  접근 svm.SVC
from sklearn.metrics import roc_auc_score, roc_curve  
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns 
#--------------------------------------------------------------------------------------------------------------------------------------
# 순서1] sns데이터셋의  titanic
df = sns.load_dataset('titanic')
print(df) 
print()
print(df.info()) 
print()

print('결측값확인')
print(df.isna().sum())
print()
print('- ' * 50)
'''
 survived  pclass     sex   age  sibsp  parch     fare embarked   class    who  adult_male deck  embark_town alive  alone
0           0       3    male  22.0      1      0   7.2500        S   Third    man        True  NaN  Southampton    no  False
1           1       1  female  38.0      1      0  71.2833        C   First  woman       False    C    Cherbourg   yes  False
2           1       3  female  26.0      0      0   7.9250        S   Third  woman       False  NaN  Southampton   yes   True
3           1       1  female  35.0      1      0  53.1000        S   First  woman       False    C  Southampton   yes  False
4           0       3    male  35.0      0      0   8.0500        S   Third    man        True  NaN  Southampton    no   True
..        ...     ...     ...   ...    ...    ...      ...      ...     ...    ...         ...  ...          ...   ...    ...
886         0       2    male  27.0      0      0  13.0000        S  Second    man        True  NaN  Southampton    no   True
887         1       1  female  19.0      0      0  30.0000        S   First  woman       False    B  Southampton   yes   True
888         0       3  female   NaN      1      2  23.4500        S   Third  woman       False  NaN  Southampton    no  False
889         1       1    male  26.0      0      0  30.0000        C   First    man        True    C    Cherbourg   yes   True
890         0       3    male  32.0      0      0   7.7500        Q   Third    man        True  NaN   Queenstown    no   True
# [891 rows x 15 columns]
 survived  pclass     sex   age  sibsp  parch     fare embarked   class    who  adult_male deck  embark_town alive  alone
'''

df_new = df.drop( ['deck', 'embark_town'] , axis=1) 
print(df_new) #[891 rows x 13 columns]
print()

df_new = df_new.dropna(subset=['age', 'embarked'],how='any', axis=0)    
print(df_new) # [712 rows x 13 columns]
print() 


# 현재 결측값은 없습니다  0/1 이라서 라벨인코더개념 
df_new = df_new[['survived', 'pclass', 'sex',  'age', 'sibsp', 'parch',  'embarked']]
print(df_new) # [712 rows x 7 columns]
print()
print(df_new.info()) # sex필드 object,  embarked필드 object
print('- ' * 50)
print()
'''
     survived  pclass     sex   age  sibsp  parch embarked
0           0       3    male  22.0      1      0        S
1           1       1  female  38.0      1      0        C
2           1       3  female  26.0      0      0        S
3           1       1  female  35.0      1      0        S
886         0       2    male  27.0      0      0        S
887         1       1  female  19.0      0      0        S
889         1       1    male  26.0      0      0        C
890         0       3    male  32.0      0      0        Q
'''

# sex성별 라벨인코딩방법
#  ㄴ첫번째 LabelEncoder화 fit_transform(),  
#  ㄴ두번째 sex_mapping={ 0:'' , 1:''}  컬럼.map(sex_mapping)
#  ㄴ세번째 pd.get_dummies()  True/False
 
sex_dummy = pd.get_dummies(df_new['sex'])
df_new = pd.concat([df_new , sex_dummy] , axis=1)
embarked_dummy = pd.get_dummies(df_new['embarked'], prefix='town')
df_new = pd.concat([df_new , embarked_dummy] , axis=1)
print(df_new) # [712 rows x 12 columns]
print()
print(df_new.info())

df_new = df_new.drop( ['sex', 'embarked'] , axis=1) 
print(df_new) # [712 rows x 10 columns]

X = df_new.iloc[ : , 1:]  # 
y =  df_new['survived']   # 생존여부
print(X)
print(y)
print()

print("X 데이터: ", X.shape) #X 데이터:  (712, 9)
print("y 데이터: ", y.shape) #y 데이터:  (712,)
print()

X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2) #, random_state=42
standard = StandardScaler() #스케일 조정 특정 열컬럼 적용할때 넘피 log
standard.fit(X_train)
x_train_scaler = standard.transform(X_train)
x_test_scaler = standard.transform(X_test)


'''
kernel = 'near', 'ploy', 'rbf기본값', 'sigmoid', 'percomputed' , 'callable') SVM 알고리즘에 사용될 커널 유형
gamma  {'scale', ' auto'} or float (default = 'scale') 커널 계수
coef0 'ploy', 'sigmoid' 에서의 독립구간
probability	boolean (default = False)'
'''

svc_model = SVC(kernel='rbf', probability=True)  #모델생성, fit(), predict(), score
svc_model.fit(x_train_scaler, y_train)      #svc_model.fit(x_train, y_train)
pred_svc = svc_model.predict(x_test_scaler) #svc_model.predict(X_test)
print('pred_svc데이터값 확인')
print(pred_svc)
print()
print('테스트정답지 데이터값 확인')
print(np.ravel(y_test))

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
score = accuracy_score(y_test, pred_svc)
print('- '  * 40)
print('score결과 비율 ' , score) #0.8037383177570093
print('score결과 비율 ' , svc_model.score(x_test_scaler, y_test)) #0.8037383177570093
print()

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test ,pred_svc)) # [[113   9]  [ 33  59]]

report = classification_report(y_test, pred_svc , labels=np.unique(pred_svc)) 
print(report)

y_train_pred = svc_model.predict(x_train_scaler) #train트레인
y_test_pred =  svc_model.predict(x_test_scaler)  ##test테스트
print('f1-score   0인경우 ', np.mean( y_train ==  y_train_pred)) #f1-score  0일때 
print('f1-score accuracy ',np.mean( y_test ==  y_test_pred))   #f1-score accuracy 
print()

# 관심분야 데이터 수집, 전처리 , 데이터분리 (X,y, test_size=0.2/0.3/0.25 )
# 모델선정, 모델생성, fit(훈련데이터), predict예측(테스트), 점수비율 
#  ㄴ모델성능 비교 , 시각화 

# 위의 내용은 03svcTitanic.py문서와 동일함  아래부분 내용 추가 
# ROC, AUC 추가


# ROC, AUC 어떻게 접근, 어떻게 처리 고민 02ROC.py문서 참고 해서 접근, 구현 

x_test_proba = svc_model.predict_proba(x_test_scaler)[ : , 1]
from sklearn.metrics import roc_curve, RocCurveDisplay  #한번 4개 다 기술해도 됩니다
fpr, tpr, thresholds = roc_curve( y_test,  x_test_proba ) #성능에서 최대 train데이터는 제외 
thresholds_index = np.arange(1, thresholds.shape[0] )

print('임계값경계값',  thresholds[thresholds_index] ) #[1 2]
print()
print(fpr[thresholds_index]) #가독성 분별력 떨어짐 , 경계영역 가늠 안됨 그래서 ROC곡선그래프 확인
print()
print(tpr[thresholds_index]) #가독성 분별력 떨어짐 , 경계영역 가늠 안됨 그래서 ROC곡선그래프 확인

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot()
RocCurveDisplay.from_estimator(svc_model, x_test_scaler, y_test, ax=ax) # AUC영역 비율 표시출력 
# plt.plot(fpr, tpr,  color='blue', label='SVC')  
plt.plot( [0,1], [0,1] , 'k--') 
plt.title('titanic 성능 ROC 그래프 확인 ')
plt.legend()
plt.show()

from sklearn.metrics import roc_auc_score, auc 
print()
print('AUC =', roc_auc_score(y_test, x_test_proba) )  
print('AUC =', auc(fpr, tpr) )

'''
tpr(True Positive Rate)는   Positive로 예측된 것들 중  Positive인것의  True 비율
tnr(True Negative Rate)는   Negative로 예측된 것들 중  Negative인것의  True 비율
fpr(False Positive Rate)는  Positive로 예측된 것들 중  Positive인것의  False 비율
fnr(False Negative Rate)는  Negative로 예측된 것들 중  Negative인것의  False 비율
'''

#pip install yadata_profiling

import ydata_profiling
pr = df.profile_report()
pr.to_file('./bike/df_report.html')


import os
import webbrowser
path = './bike/sweetviz_report_df.html'
webbrowser.open(os.path.realpath(path))

print(path, '파일 오픈 실행까지 확인했습니다')




print()
print()