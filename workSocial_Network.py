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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
#--------------------------------------------------------------------------------------------
# 해결1 Social_Network_Ads.csv파일 읽기 , 정보확인info, 컬럼, 널값, 구조 등등 확인
# 해결2 from sklearn.model_selection import train_test_split(훈련데이터, 정답지, 분리테스트2/훈련8)
  #ㄴ보충 train_data대신 X,  target대신y
# 해결3 train_data대신 X훈련데이터  MinMaxScaler적용 
# 해결4 train_test_split() 데이터분리 훈련,테스트 분리 
# 해결5 학습모델선택, fit(), predict()결과는 [0 0 0 1 0 1 0 0 0  ~ ~  0 1 0 1 1 0 0 0  1 0 0 1 1 0]  

# from sklearn.metrics import confusion_matrix 
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import classification_report
  #ㄴ 해결6  confusion_matrix
  #ㄴ 해결6  accuracy_score
  #ㄴ 해결6  classification_report구매/비구매 전체데이터중 구매% 비구매%  시각화표현 [0 0 0 1 0 1 0 0 0  ~ ~  0 1 0 1 1 0 0 0  1 0 0 1 1 0]

df = pd.read_csv('./data/Social_Network_Ads.csv')
print(df.info())
'''
 0   User ID          400 non-null    int64 
 1   Gender           400 non-null    object
 2   Age              400 non-null    int64
 3   EstimatedSalary  400 non-null    int64
 4   Purchased        400 non-null    int64
'''
print()
print(df)  #[400 rows x 5 columns]
time.sleep(1)


X = df.loc[ : , 'Age':'EstimatedSalary']  #훈련데이터 train_data맞아요, 범위차이가 많이 나네요  MinMaxScaler화
y = df['Purchased']   #정답지 target

# print('MinMaxScaler 스케일조정 전 ')
# print(X)
# print()

print('MinMaxScaler 스케일조정 후 ')
# scaler_x = MinMaxScaler() # 0 ~ 1
scaler_x = StandardScaler() # -1 ~ 1사이 
X = scaler_x.fit_transform(X)
# print()
# print(X)
# print()

#훈련데이터, 테스트데이터 분리  8훈련:2테스트  2테스트:8훈련
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2 )  # , random_state = 1

#훈련처리하는 모델선정  구매를 할까 말까   분류 문제  LogisticRegression, fit(훈련X,훈련y) 
model = LogisticRegression()
model.fit(X_train, y_train)
print('모델의 fit()처리 종료\n')
y_pred = model.predict(X_test) 
print('예측 데이터 출력')
print(y_pred)  #[0 0 0 1 0 1 0 0 0  ~ ~  0 1 0 1 1 0 0 0  1 0 0 1 1 0]

# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import classification_report
print('\nfrom sklearn.metrics import confusion_matrix함수')
cm = confusion_matrix(y_test ,y_pred)
print(cm)   # [52,  6],  [14, 28]]

print('\nfrom sklearn.metrics import accuracy_score함수')
score = accuracy_score(y_test, y_pred)
print('점수score =' , score)

print('\nfrom sklearn.metrics import classification_report함수')
report = classification_report(y_test, y_pred)
print(report)

'''
from sklearn.metrics import confusion_matrix함수
[[52,  6],  [14, 28]]

from sklearn.metrics import accuracy_score함수  
점수score = 0.80

classification_report   , ROC커브 = Receiver Operating Characteristic" 커브로 binary classification 모델
ROC커브 = Receiver Operating Characteristic" 커브로 binary classification 모델
AUC는 "Area Under  Curve "  ROC 커브 아래 부분의 면적의 너비를 말합니다.
Sensitivity : tpr=TPR(True Positive Rate)는 Positive로 예측된 것들 중 진짜 Positive인 것의 비율을 나타냅니다. ( TPR = TP / (TP+FN) )
Specificity : tnr=TNR(True Negative Rate)는 Negative로 예측된 것들 중 진짜 Negative인 것의 비율을 나타냅니다. (TNF = TN / (TN + FP)) 

              precision    recall    f1-score   support

           0       0.79      0.90      0.84        58
           1       0.82      0.67      0.74        42

    accuracy                           0.80       100
   macro avg       0.81      0.78      0.79       100
weighted avg       0.80      0.80      0.80       100
'''
print('- ' * 50)


'''
np.meshgrid(xi, copy=True, sparse=False, indexing='xy')
xi : 그리드에 나타낼 1차원 배열 벡터
copy : 디폴트는 True, False가 되면 메모리 절약을 위해 오리지널 배열로 반환
sparse : 디폴트는 False, True로 하면 메모리 절약을 위해 희소 그리드를 반환
indexing : {'xy','ij}, 카테시안 인덱스(일반좌표) 표시를 원하면 'xy', 행렬 인덱스는 'ij'
'''

from matplotlib.colors import ListedColormap
x_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))


print(df) #[400 rows x 5 columns]
print()    


plt.figure(figsize=(10,6))  #여러분 12, 7 
plt.contourf(X1, X2, model.predict( np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),alpha=0.75, cmap=ListedColormap(('hotpink','green') ))

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0] ,  x_set[y_set==j,1] ,  c=ListedColormap(('red','green'))(i), label=j)
    # plt.scatter(x_set[y_set==j,0] ,  x_set[y_set==j,1] ,  c='green', label=j)

plt.title('classification_report 차트로 확인')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()
 












print()
print()
