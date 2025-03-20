import time
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
from matplotlib import font_manager
from sklearn.ensemble import RandomForestRegressor # 랜덤 포레스트 회귀
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import skew, kurtosis # 첨도, 왜도

mpl.rc('axes', unicode_minus=False)
mpl.rcParams['axes.unicode_minus'] = False

font_name = font_manager.FontProperties(fname='c:/windows/Fonts/malgun.ttf').get_name()
rc('font', family=font_name)

train = pd.read_csv('./bike/train.csv')
test = pd.read_csv('./bike/test.csv')

'''
# train 데이터에서 1000행까지만 선택
# train = train.head(1000)
# train = train[:1000]
둘 중 하나 선택
'''

print(train)
print()
print(test)

# train에 있는 훈련 데이터에 있는 count만 추출
train_count_col = train['count']
print(train_count_col)
print()

# count 열의 왜도 및 첨도 측정
count_skewness = skew(train['count']) # 왜도
count_kurtosis = kurtosis(train['count']) # 첨도

print(f"Count 열의 왜도: {count_skewness}")
print(f"Count 열의 첨도: {count_kurtosis}")

start_time = time.time() # 시간 측정 시작

# count 열의 분포 시각화 및 왜도, 첨도 표시
plt.figure(figsize=(12, 7))
sns.distplot(train['count'], kde=True)
plt.title(f"Count Distribution (Skewness: {count_skewness:.2f}, Kurtosis: {count_kurtosis:.2f})")
plt.xlabel("Count")
plt.ylabel("Density")

plt.tight_layout()
plt.show()


# 날씨 카테고리별 개수 출력
weather_counts = train['weather'].value_counts()
print(weather_counts)
print()

# 계절 카테고리별 개수 출력
season_counts = train['season'].value_counts()
print(season_counts) 

# 데이터 전처리
def preprocess_data(df):
    # 시간 데이터 처리
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year.astype(np.float32)
    df['quarter'] = df['datetime'].dt.quarter.astype(np.float32)
    df['month'] = df['datetime'].dt.month.astype(np.float32)
    df['day'] = df['datetime'].dt.day.astype(np.float32)
    df['hour'] = df['datetime'].dt.hour.astype(np.float32)
    df['weekday'] = df['datetime'].dt.weekday.astype(np.float32)
    
    df = df.drop('datetime', axis=1)
    
    if 'casual' in df.columns:
        df = df.drop(['casual', 'registered'], axis=1)
    
    numerical_cols = ['temp', 'atemp', 'humidity', 'windspeed']
    for col in numerical_cols:
        df[col] = np.log1p(df[col]).astype(np.float32)
        
    return df

train = preprocess_data(train)
test = preprocess_data(test)

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

X_train = train.drop('count', axis=1).astype(np.float32)
y_train = train['count'].astype(np.float32)
X_test = test.astype(np.float32)

rf_reg = RandomForestRegressor(
    n_estimators=500, # 트리 수
    max_depth=15, # 트리 수
    min_samples_split=5, # 트리 수
    min_samples_leaf=1, # 최소 리프 노드(제일 끝 노드) 샘플 수
    max_features='sqrt', # 최대 특성 수 
    bootstrap=True, # 부트스트랩 샘플링 (True면 중복을 허용)
    random_state=42 # 난수 시드 설정 값
)

rf_reg.fit(X_train, y_train)

y_pred = rf_reg.predict(X_test)

if 'count' in train.columns:
    y_train_pred = rf_reg.predict(X_train)
    mse = mean_squared_error(y_train, y_train_pred)
    r2 = r2_score(y_train, y_train_pred)
    print(f'Train Data RMSE: {np.sqrt(mse)}')
    print(f'Train Data R²: {r2}')

print()
submission = pd.DataFrame({'datetime': pd.read_csv('./bike/test.csv')['datetime'],
                           'count': np.expm1(y_pred)})
submission['count'] = submission['count'].apply(lambda x: 0 if x < 0 else x)
submission.to_csv('submission.csv', index=False)

# 시각화

# 플롯 저장 디렉터리 설정
plots_dir = ('./plots')
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
    
# 특성 중요도 시각화
feature_importances = pd.Series(rf_reg.feature_importances_, index=X_train.columns)
feature_importances = feature_importances.sort_values(ascending=False)
plt.figure(figsize=(12, 7))
sns.barplot(x=feature_importances, y=feature_importances.index, palette='pastel')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance')
plt.tight_layout()

plt.savefig(os.path.join(plots_dir,'plots1_feature_importance.png'))
plt.show()

# 시간별 자전거 대여량 변화(연도, 월, 시간)
plt.figure(figsize=(22, 20))
plt.subplot(2, 2, 1)
sns.boxplot(x='year', y='count', data=train, palette='pastel')
plt.title('Bike Rentals by Year')
plt.xlabel('Year')
plt.ylabel('Rental Count')
plt.xticks(rotation=45, ha='right')

plt.subplot(2, 2, 2)
sns.boxplot(x='quarter', y='count', data=train, palette='pastel')
plt.title('Bike Rentals by Quarter')
plt.xlabel('Year')
plt.ylabel('Rental Count')
plt.xticks(rotation=45, ha='right')

plt.subplot(2, 2, 3)
sns.boxplot(x='month', y='count', data=train, palette='pastel')
plt.title('Bike Rentals by Month')
plt.xlabel('Month')
plt.ylabel('Rental Count')
plt.xticks(rotation=45, ha='right')

plt.subplot(2, 2, 4)
sns.boxplot(x='hour', y='count', data=train, palette='pastel')
plt.title('Bike Rentals by Hour')
plt.xlabel('Hour')
plt.ylabel('Rental Count')
plt.xticks(rotation=45, ha='right')

plt.savefig(os.path.join(plots_dir,'plots2_datetime.png'))
plt.show()

#  요일별 자전거 대여량 비교
plt.figure(figsize=(12, 7))
sns.boxplot(x='weekday', y='count', data=train, palette='pastel')
plt.title('Bike Rentals by Weekday')
plt.xlabel('Weekday (0: Monday, 1: Tuesday, ..., 6: Sunday)')
plt.ylabel('Rental Count')
plt.tight_layout()

plt.savefig(os.path.join(plots_dir,'plots3_weekdays.png'))
plt.show()

# 날씨별 자전거 대여량 비교
plt.figure(figsize=(12, 7))
sns.boxplot(x='weather', y='count', data=train, palette='pastel')
plt.title('Bike Rentals by Weather')
plt.xlabel('Weather (1: Clear, 2: Mist, 3: Light Snow/Rain, 4: Heavy Snow/Rain)')
plt.ylabel('Rental Count')
plt.tight_layout()

plt.savefig(os.path.join(plots_dir,'plots4_weather.png'))
plt.show()

# 계절별 자전거 대여량 비교
plt.figure(figsize=(12, 7))
sns.boxplot(x='season', y='count', data=train, palette='pastel')
plt.title('Bike Rentals by Season')
plt.xlabel('Season (1: Spring, 2: Summer, 3: Fall, 4: Winter)')
plt.ylabel('Rental Count')
plt.tight_layout()

plt.savefig(os.path.join(plots_dir, 'plots5_season.png'))
plt.show()

# 온도 및 체감 온도에 따른 자전거 대여량 비교
plt.figure(figsize=(12, 7))
plt.subplot(1,2,1)
sns.scatterplot(x='temp', y='count', data=train, palette='pastel')
plt.title('Bike Rentals by Temperature')
plt.xlabel('Temperature')
plt.ylabel('Rental Count')
plt.legend()

plt.subplot(1, 2, 2)
sns.scatterplot(x='temp', y='count', data=train, palette='pastel')
plt.title('Bike Rentals by Apparent Temperature')
plt.xlabel('Apparent Temperature')
plt.ylabel('Rental Count')
plt.tight_layout()

plt.savefig(os.path.join(plots_dir,'plots6_temperature.png'))
plt.show()

# 습도, 풍속과 자전거 대여량 비교
plt.figure(figsize=(12, 7))
plt.subplot(1,2,1)
sns.scatterplot(x='humidity', y='count', data=train, palette='pastel')
plt.title('Bike Rentals by Humidity')
plt.xlabel('Humidity')
plt.ylabel('Rental Count')

plt.subplot(1, 2, 2)
sns.scatterplot(x='windspeed', y='count', data=train, palette='pastel')
plt.title('Bike Rentals by Windspeed')
plt.xlabel('Windspeed')
plt.ylabel('Rental Count')
plt.tight_layout()

plt.savefig(os.path.join(plots_dir,'plots7_humid_and_windspd.png'))
plt.show()

print()
end_time=time.time() # 측정 종료료
print('start_time : ', start_time)
print('end_time : ', end_time)
print()
print('Total time taken to generate all plots', start_time - end_time) # 총 걸린 시간

# 


