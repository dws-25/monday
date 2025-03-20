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


'''
# ROC커브 = Receiver Operating Characteristic" 커브로 binary classification 모델의 성능를 나타내는 그래프입니다
#그래프의 X축은 False Positive Rate(FPR) Y축은 True Positive Rate(TPR)을 나타냅니다.
# AUC는 "Area Under  Curve "  ROC 커브 아래 부분의 면적의 너비를 말합니다.
Sensitivity:  tpr=TPR(True Positive Rate)는  Positive로 예측된 것들 중 진짜 Positive인 것의 비율을 나타냅니다. ( TPR = TP / (TP+FN) )
Specificity : tnr=TNR(True Negative Rate)는  Negative로 예측된 것들 중 진짜 Negative인 것의 비율을 나타냅니다. (TNF = TN / (TN + FP))                           
'''
# 사이킷럿(scikit-learn) 라이브러리 임포트 
from sklearn.datasets import make_classification 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, roc_curve  #처음적는 함수 
from sklearn.model_selection import train_test_split
#--------------------------------------------------------------------------------------------------------------------------------------
# DecisionTreeClassifier,  LogisticRegression,  SVC 성능 비교해서 ROC커브 그래프 표시
# 순서1] 데이터발생 혹은 열기 
X,y = make_classification(n_samples=1000, n_classes=2 ) #, random_state=42
print('X데이터')
print(X)
print()
print('y데이터')  #구매1/비구매0, 생존여부1/비생존0
print(y)
print()
 
# 순서2] 데이터분리  from sklearn.model_selection import train_test_split  , random_state=42
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.3)

# 순서3] 모델생성  dt, lr, svc / fit(훈련x,훈련y) / 예측 모델.predict(X_test) ===>세트 
print('모델생성및 fit(훈련x, 훈련y)함수처리')

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

lr = LogisticRegression()
lr.fit(X_train, y_train)

svc = SVC(probability=True) 
svc.fit(X_train, y_train)

print('predict()함수처리')
y_pre_dt = dt.predict_proba(X_test) #predict_proba()예측보다는 성능의 비교, 임계=경계threshold
y_pre_lr = lr.predict_proba(X_test)
y_pre_svc = svc.predict_proba(X_test)
print('- ' * 35)

# 순서4] score, accuracy  데이터생성, 데이터훈련테스트분리, 모델생성/모델fit()/predict(x_test)
# from sklearn.metrics import roc_auc_score, roc_curve  #처음적는 함수 
print('roc_curve()함수처리')
fpr_dt, tpr_dt, z = roc_curve(y_test, y_pre_dt[: , 1])
fpr_lr, tpr_lr, z  = roc_curve(y_test, y_pre_lr[: , 1])
fpr_svc, tpr_svc, z  = roc_curve(y_test, y_pre_svc[: , 1])
print('2시 07분 fpr_svc, tpr_svc, z =', z)

plt.figure(figsize=(10,7))
plt.plot(fpr_dt, tpr_dt,  color='red', label='DecisionTree') 
plt.plot(fpr_lr, tpr_lr,  color='green', label='LogisticRegression') 
plt.plot(fpr_svc, tpr_svc,  color='blue', label='SVC') 
plt.plot( [0,1], [0,1] , 'k--')

plt.title('ROC커브 그래프')
plt.xlabel('fpr Fale Positive Rate')
plt.ylabel('tpr True Positive Rate')
plt.legend()
plt.show()

print('- ' * 50)
print()
# 순서5]  AUC는 "Area Under  Curve "  ROC 커브 아래 부분의 면적의 너비
# from sklearn.metrics import roc_auc_score, roc_curve  #처음적는 함수 
auc_dt = roc_auc_score(y_test, y_pre_dt[: , 1])
auc_lr = roc_auc_score(y_test, y_pre_lr[: , 1])
auc_svc = roc_auc_score(y_test, y_pre_svc[: , 1])
print('AUC DecisionTreeClassifier() 면적 ', auc_dt )
print('AUC LogisticRegression() 면적 ', auc_lr )
print('AUC SVC(probability=True)  면적 ', auc_svc )

'''
# ROC커브 = Receiver Operating Characteristic" 커브로 binary classification 모델의 퍼포먼스를 나타내는 그래프입니다
#그래프의 X축은 False Positive Rate(FPR) Y축은 True Positive Rate(TPR)을 나타냅니다.
# AUC는 "Area Under  Curve "  ROC 커브 아래 부분의 면적의 너비를 말합니다.
Sensitivity:  tpr=TPR(True Positive Rate)는  Positive로 예측된 것들 중 진짜 Positive인 것의 비율을 나타냅니다. ( TPR = TP / (TP+FN) )
Specificity : tnr=TNR(True Negative Rate)는  Negative로 예측된 것들 중 진짜 Negative인 것의 비율을 나타냅니다. (TNF = TN / (TN + FP))                           
'''

# AUC DecisionTreeClassifier() 면적  0.8372155049786628
# AUC LogisticRegression()     면적  0.9408339260312945
# AUC SVC(probability=True)    면적  0.9231863442389758

# AUC DecisionTreeClassifier() 면적  0.8092718794835008
# AUC LogisticRegression()     면적  0.9363342898134864
# AUC SVC(probability=True)    면적  0.9333751793400287










print()
print()