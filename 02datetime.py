
import pandas as pd
import numpy as np
import time

#-----------------------------------------------------------------------------------------------------
data = {
    'date': ['2025-02-25', '2025-06-24', '2025-02-23', '2025-03-15', '2025-03-16'],
    'sales': [100, 150, 200, 250, 300]
}

# 해결1] 판다스의 DataFrame화 df = pd.DataFrame()
#         info() 각각필드 정보 다시한번 확인 
# 해결2] 'date'컬럼 날짜화  pd.to_datetime()
# 해결3]  날짜변환후 dt컴바인속성 접근 월추출month, 요일추출dayofweek, 분기추출 quarter
# 해결4]  요일추출하면 숫자출력 
df = pd.DataFrame(data)
print(df.info())  # date    5 non-null      object
print()

df['date'] = pd.to_datetime(df['date']) 
print(df.info())  #  date    5 non-null      datetime64[ns]
print()
print(df)
print()

df['year'] = df['data'].dt.year
df['month'] = df['date'].dt.month
# df['dayofweek'] = df['date'].dt.day_name()
df['dayofweek'] = df['date'].dt.day_of_week
df['quarter'] = df['date'].dt.quarter
# 월요일0  일요일6

print(df)
print()























print()
print()