
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
'''
사이킷런은 fit()과 transform()을 하나로 결합한 fit_transform()도 제공합니다.

모델선정
학습: fit()
예측: predict()
가장 유명한 것은 K-means 클러스터링  주로 고객을 그룹화하는 등 공통된 특성을 가진 군집들로 묶는 알고리즘을 의미합니다.
https://www.kaggle.com/code/yutohisamatsu/customer-uselog-practice2-python-book
'''

ul = pd.read_csv('./bike/use_log.csv'  )
cj = pd.read_csv('./bike/customer_join.csv') #비지도학습  정답이 없기떄문에 비지도학습

print(ul)  # log_id customer_id     usedate
print()    # [197428 rows x 3 columns]

print(cj)  
# customer_id  name class gender  start_date end_date campaign_id is_deleted class_name price  
# campaign_name  mean median  max min  routine_flg  calc_date  membership_period
print() #[4192 rows x 18 columns]

print(ul.info())  #  [197428 rows x 3 columns]
print()

print(cj.info())   #  #[4192 rows x 18 columns]
print('- ' * 40)
print()
print()

'''
가장 전통적 방법인 k-means 클러스터링 실시
<조건>
1. 그룹 개수 설정: 4개
2. mean~min은 월 이용 횟수와 관련한 변수이므로 1~8 사이값을 갖지만, membership_period는 이에 비해 값이 너무 크다.
3. membership_period에는 표준화를 하자.

고객별 한달 이용 이력 데이터인 mean, median, max, min, mem_period(즉, 각 고객별 한달의 평균, 중위수, 최대, 최소 이용횟수 및 회원기간(단위: month))
5개 변수의 값을 평균 0, 표준편차 1을 따르는 정규분포를 따르도록 바꿔줍니다. 이를 '표준화'라고 합니다.
'''
cc = cj[['mean', 'median', 'max', 'min', 'membership_period']]
print(cc) #[4192 rows x 5 columns]
print()

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#표준화
sc = StandardScaler()
cc_scaled = sc.fit_transform(cc) # cc = cj[['mean', 'median', 'max', 'min', 'membership_period']]
df = pd.DataFrame(cc_scaled)    # 모든변수의 값이 비슷해짐 
print( df ) #[4192 rows x 5 columns]
print('- ' * 40)
print()


kmeans = KMeans(n_clusters=4, random_state=0)
clusters = kmeans.fit(cc_scaled)
print(clusters) # KMeans(n_clusters=4, random_state=0)
print()


cc['cluster'] = clusters.labels_
print(cc['cluster'].unique()) # cluster가 0, 1, 2, 3 이렇게 4개가 생성 그룹으로 군집화 완료! (클러스터 넘버: 0, 1, 2, 3)
print(cc.head())  
print()

#cc = cj[['mean', 'median', 'max', 'min', 'membership_period']] 한글화
cc.rename(columns = {'mean':'월평균값', 'median':'월중앙값', 'max':'월최대값', 'min':'월최소값', 'membership_period':'회원기간'}, inplace=True)
print(cc) #[4192 rows x 6 columns]
print()

print(cc.groupby('cluster').count()) #  그룹0에 해당 갯수가 1334제일많음
print()
'''
그룹 0에 해당하는 수가 1334개로 가장 많으며, 그 다음이 그룹3 > 그룹2 > 그룹1 순
월평균값  월중앙값  월최대값  월최소값  회원기간 
0        1334       1334     1334  1334  1334
1         771       771     771    771   771
2        1249       1249    1249   1249  1249
3         838       838     838     838   838
'''

print(cc.groupby('cluster').mean())
print()
'''
         월평균값      월중앙값       월최대값        월최소값       회원기간 
0        5.541974       5.392804     8.757871        2.704648       14.857571
1        3.065504       2.900130     4.783398        1.649805       9.276265
2        4.677561       4.670937     7.233787       2.153723        36.915933
3        8.064079       8.050716     10.014320       6.180191       7.016706

그룹 3는 평균 회원기간은 가장 짧지만(7일), 월평균 이용횟수는 8회로 가장 높네요.
그룹 1은 평균 회원기간도 9일로 짧은데다가 월평균 이용횟수는 3회로 가장 낮습니다.
'''


'''
클러스터링 결과 시각화(차원 축소 활용)
사용한 변수는 5개로, 5차원 그래프는 그리기도 힘들뿐더러 이해하기도 쉽지 않습니다.
우리는 5개의 변수를 2개의 변수로 줄이는 작업을 통해, 2차원 그래프로 나타내봅니다. 이것이 바로 '차원 축소'인데요,
차원축소의 대표적 방법인 주성분분석(PCA=Principal Component Analysis)=자원축소
애매모호하죠 어떤변수를 선택했는지를 ... 
'''
from sklearn.decomposition import PCA
X = cc_scaled.copy()
pca = PCA(n_components=2)
pca.fit(X)
x_pca = pca.transform(X)
print(x_pca)
print('- ' * 40)
print()



pca_df = pd.DataFrame(x_pca)
pca_df['cluster'] = cc['cluster'] 
print("pca_df.head() 데이터 확인")  #  0  1  cluster 생성 
print(pca_df.head())
print()


sorted(pca_df['cluster'].unique())
sns.scatterplot(x=0, y=1,  data=pca_df, hue='cluster') # marker='+'   s=50, color='r'
plt.show()
print()

#plt.scatter더 효과적 선명도 뛰어남
for i in sorted(pca_df['cluster'].unique()):
    tmp = pca_df.loc[pca_df['cluster'] == i]
    plt.scatter(tmp[0], tmp[1])
    plt.legend(sorted(pca_df['cluster'].unique()))

plt.show()
print()


# 클러스터링 결과를 바탕으로, 탈퇴 회원의 경향을 파악
cc_scaled = sc.fit_transform(cc) 
cc['cluster'] = clusters.labels_
print(cc['cluster'].unique()) # cluster가 0, 1, 2, 3 이렇게 4개가 생성 그룹으로 군집화 완료! (클러스터 넘버: 0, 1, 2, 3)
print(cc) 
print()

# cc에서 지속/탈퇴회원 여부를 알아야 하므로 'is_deleted' 열을 추가한다. (이 열은 cj 데이터에 있으므로 둘을 조인)
print("cc_join 연결  = cc + cj ")
cc_join= pd.concat([cc, cj], axis=1) 
print(cc_join) 
print()


# 우리가 필요한 것은 클러스터별 탈퇴여부이므로, 이에 대해 groupby를 해서 해당 회원이 몇 명이나 있는지 알아봅시다.
new_df = cc_join.groupby(['cluster','is_deleted'], as_index=False).count()[['cluster', 'is_deleted', 'customer_id']]
print(new_df)
print()


# new_df에서 cluster별 탈퇴 및 미탈퇴회원 비율을 for문으로 뽑아보자.
de0 = (new_df['is_deleted']==0) #지속회원
de1 = (new_df['is_deleted']==1) #탈퇴회원










print()
print()