# import matplotlib
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
#--------------------------------------------------------------------------------------------
df =  sns.load_dataset('tips')
print(df)  ## 처음데이터 [244 rows x 7 columns]
print(df.info()) 
print()

# sns.countplot(x='day', data=df, palette='Set1')
# plt.show()

# sns.barplot(x='sex', y='tip', data=df, palette='Set2')
# plt.show()


# sns.boxplot(x='time', y='total_bill', data=df, palette='Set3')
# plt.show()

# sns.histplot( x='total_bill', data=df, palette='Set1')
# plt.show()

# sns.scatterplot( x='total_bill', y='tip', data=df,  c='green')
# plt.show()


mean = np.mean(df['total_bill'])
std = np.std(df['total_bill'])

print('총금액 평균 ', mean)
print('총금액 표편 ',std)
upper_limit = mean + 3*std   # 46.43839435626422
lower_limit = mean - 3*std   # -6.866509110362578
print('upper_limit ' , upper_limit)
print('lower_limit ' , lower_limit)
print()


Q1 = df['total_bill'].quantile(0.25)
Q3 = df['total_bill'].quantile(0.75)

IQR = Q3 - Q1
print('IQR =', IQR)
uppper_limit = Q3 + 1.5 * IQR   # 46.43839435626422
lower_limit = Q1 - 1.5 * IQR    # -2.8224999999999945
print('upper_limit ' , upper_limit)
print('lower_limit ' , lower_limit)
print()


# # 처음데이터 [244 rows x 7 columns]
# 이상치 조건 적용 
cond = (df['total_bill'] > upper_limit) | (df['total_bill'] < lower_limit)

# # 데이터프레임[boolean]을 넣으면, True인 값만 나옴! 
print(df[cond])
print()

df = df[~cond]
print(df[55:65])
print(df[165:175])
print()
# # 처음데이터 [244 rows x 7 columns]


