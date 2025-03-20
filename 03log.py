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
#--------------------------------------------------------------------------------------------


num = 8
print('원래 num =', num)
print("log(8) =", np.log(num))      # 자연로그  2.0794415416798357
print("log10(8) =", np.log10(num)) # 밑이 10인 로그  0.9030899869919435
print("log2(8) =", np.log2(num))   # 밑이 2인 로그   3.0
print()

x = 10
log_x = np.log(x)
print(f"ln({x}) =", log_x)  # 2.302585092994046
print()

# 배열
arr = np.array([1, 2, 4, 10, 100, 1000])  #10숫자  ~ 100숫자 크기 
log_arr = np.log(arr) 
print("log(arr) =", log_arr) # [0.    0.69314718 1.38629436 2.30258509 4.60517019  6.90775528 ]

log_arr = np.log2(arr)
print("log(arr) =", log_arr)

print()
print()