# %% 앙상블 모형 기반 신용카드 일별 매출 예측
# %% 0. Environment Settings
import platform
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use('ggplot')

if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
else:
    plt.rc('font', family='AppleGothic')
plt.rc('axes', unicode_minus=False)

# %% 1. LOAD THE DATA
# Data Source: 금융데이터거래소 - [NH농협카드] 일자별 소비현황_서울
bas_ym = pd.date_range(start='20200101', end='20240131', freq='MS').strftime('%Y%m').tolist()

df = pd.DataFrame()

for i, var in enumerate(bas_ym):
    data_path = f'data/[NH농협카드] 일자별 소비현황_서울_{var}.csv'
    
    # Read the data with available encodings
    encodings = ['utf-8-sig', 'euc-kr', 'cp949']
    for encoding in encodings:
        try:
            tmp_df = pd.read_csv(data_path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Failed to read {data_path} with available encodings.")
    
    df = pd.concat([df, tmp_df], axis=0)
    
print(df.shape)
print(df.head())


# %% 2. DATA PREPROCESSING
# Type Conversion (int64 -> datetime64)
df['date'] = pd.to_datetime(df['승인일자'], format='%Y%m%d')

# Decimal Point Handling
df['이용금액_전체_억원'] = df['이용금액_전체'] / 100
df['이용금액_개인_억원'] = df['이용금액_개인'] / 100
df['이용금액_법인_억원'] = df['이용금액_법인'] / 100

# Derived Variables
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek

# Nominal to Ordinal Variable
df['dayname'] = pd.Categorical(df['date'].dt.day_name(), 
                               categories=['Monday', 'Tuesday', 'Wednesday','Thursday', 'Friday', 'Saturday', 'Sunday'],
                               ordered=True)

# Add weekend variable
df['weekend'] = df['dayname'].isin(['Saturday', 'Sunday'])

# Nullity Check
print(df.isna().sum())

# Train-Test Split
df_train = df[df['year'] != 2024]
df_test = df[df['year'] == 2024]

print(df_train.shape, df_test.shape)

# Reset Index
df.reset_index(drop=True, inplace=True)


# %% 3. EDA
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,6))

# 3-1. Null Handling
sns.heatmap(df.isna(), ax=axes[0])
axes[0].set_title('결측치 NULL 확인')
axes[0].set_yticklabels([])

# 3-2. Correlation Matrix (Multicolinearity Check)
corr_mat = df.drop(['시도', '승인일자', 'dayname'], axis=1).corr()

sns.heatmap(corr_mat, annot=True, fmt='.1f', ax=axes[1])
axes[1].set_yticklabels([])

axes[1].set_title('변수 간 상관관계')
plt.tight_layout()
plt.show()

# %% 3-3. Outlier Detection
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12,9))

variables = ['이용금액_전체_억원', '이용금액_개인_억원', '이용금액_법인_억원']
 
for i, vars in enumerate(variables):
    sns.lineplot(data=df, x='date', y=vars,
                ax=axes[i%3][0])
    sns.boxplot(data=df, x=vars,
                ax=axes[i%3][1])
    axes[i%3][0].set_title(vars)
    axes[i%3][0].tick_params(axis='x', rotation=30)

plt.tight_layout()
fig.suptitle('')
plt.show()

# %% 3-4. Sales by Day & Month
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,8))

# Row 1: 요일별
sns.barplot(data=df, x='dayname', y='이용건수_전체',
            hue='dayname', palette='crest', legend=False, ax=axes[0][0])
sns.barplot(data=df, x='dayname', y='이용금액_전체_억원',
            hue='dayname', palette='crest', legend=False, ax=axes[0][1])
axes[0][0].set_title('요일별 이용건수')
axes[0][1].set_title('요일별 이용금액 (억원)')

# Row 2: 월별
sns.lineplot(data=df, x='month', y='이용건수_전체', ax=axes[1][0])
sns.lineplot(data=df, x='month', y='이용금액_전체_억원', ax=axes[1][1])
axes[1][0].set_title('월별 이용건수')
axes[1][1].set_title('월별 이용금액 (억원)')
axes[1][0].set_xticks(range(1,13))
axes[1][1].set_xticks(range(1,13))

fig.suptitle('신용카드 이용 현황 (요일별 / 월별)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# %% 4. 모델링
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

# 4-1. Time Window Expanding Cross Validation
ind_vars   = ['승인일자', 'year', 'month', 'day', 'dayofweek']
dep_var    = '이용금액_개인_억원'
WEIGHT_MAP = {2020: 0.1, 2021: 0.2, 2022: 0.3, 2023: 0.4}

df_cv_data = df[df['year'] != 2024].copy()
periods = sorted(df_cv_data['date'].dt.to_period('M').unique())

cv_results = []

for i in range(1, len(periods)):
    train_mask = df_cv_data['date'].dt.to_period('M').isin(periods[:i])
    test_mask  = df_cv_data['date'].dt.to_period('M') == periods[i]

    X_tr = df_cv_data[train_mask][ind_vars]
    y_tr = df_cv_data[train_mask][dep_var]
    X_te = df_cv_data[test_mask][ind_vars]
    y_te = df_cv_data[test_mask][dep_var]
    w_tr = df_cv_data[train_mask]['year'].map(WEIGHT_MAP).values

    m_rf   = RandomForestRegressor(
        n_estimators=200, max_depth=5, max_features='sqrt', min_samples_leaf=3, random_state=42)
    m_xgb  = XGBRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=5, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=3, reg_alpha=0.05, random_state=42)
    m_lgbm = LGBMRegressor(
        n_estimators=200, learning_rate=0.05, num_leaves=31, max_depth=5,
        subsample=0.8, colsample_bytree=1.0, min_child_samples=20, reg_alpha=0,
        random_state=42, verbose=-1)

    m_rf.fit(X_tr, y_tr, sample_weight=w_tr)
    m_xgb.fit(X_tr, y_tr, sample_weight=w_tr)
    m_lgbm.fit(X_tr, y_tr, sample_weight=w_tr)

    cv_results.append({
        'test_period': str(periods[i]),
        'train_months': i,
        'mse_rf':   mean_squared_error(y_te, m_rf.predict(X_te)),
        'mse_xgb':  mean_squared_error(y_te, m_xgb.predict(X_te)),
        'mse_lgbm': mean_squared_error(y_te, m_lgbm.predict(X_te)),
    })

df_cv = pd.DataFrame(cv_results)
print(df_cv.to_string(index=False))

# 4-2. Machine Learning (RandomForest, XGBoost, LightGBM) - 전체 훈련데이터로 최종 학습
X_train = df_cv_data[ind_vars]
y_train = df_cv_data[dep_var]

w_all = df_cv_data['year'].map(WEIGHT_MAP).values

model_rf   = RandomForestRegressor(
    n_estimators=1000, max_depth=5, max_features='sqrt', min_samples_leaf=3, random_state=42)
model_xgb  = XGBRegressor(
    n_estimators=1000, learning_rate=0.05, max_depth=5, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=3, reg_alpha=0.05, random_state=42)
model_lgbm = LGBMRegressor(
    n_estimators=1000, learning_rate=0.05, num_leaves=31, max_depth=5,
    subsample=0.8, colsample_bytree=1.0, min_child_samples=20, reg_alpha=0,
    random_state=42, verbose=-1)

model_rf.fit(X_train, y_train, sample_weight=w_all)
model_xgb.fit(X_train, y_train, sample_weight=w_all)
model_lgbm.fit(X_train, y_train, sample_weight=w_all)

# 4-3. CV 결과 요약
print(f'Mean MSE (RandomForest): {df_cv["mse_rf"].mean():.2f}')
print(f'Mean MSE (XGBoost):      {df_cv["mse_xgb"].mean():.2f}')
print(f'Mean MSE (LightGBM):     {df_cv["mse_lgbm"].mean():.2f}')

best_model = model_lgbm

# %% 5. 결과
# 5-1. Model Evaluation
test_period = df_test['date'].between('2024/01/01', '2024/01/31')

x_range = np.arange(1,len(df_test[test_period])+1)
X_test = df_test[test_period][ind_vars]
y_test = df_test[test_period][dep_var]

y_pred_rf = model_rf.predict(X_test)
y_pred_xgb = model_xgb.predict(X_test)
y_pred_lgbm = model_lgbm.predict(X_test)

mse_rf = mean_squared_error(y_pred_rf, y_test)
mse_xgb = mean_squared_error(y_pred_xgb, y_test)
mse_lgbm = mean_squared_error(y_pred_lgbm, y_test)

print(f'MSE (RandomForest): {mse_rf:.2f}')
print(f'MSE (XGBoost): {mse_xgb:.2f}')
print(f'MSE (LightGBM): {mse_lgbm:.2f}')

# ### 비즈니스 관점에서의 선택
print(f'''
RandomForest 예측 오차   : {y_pred_rf.sum() - y_test.sum():.2f} 억원
XGBoost 예측 오차        : {y_pred_xgb.sum() - y_test.sum():.2f} 억원
LightGBM 예측 오차       : {y_pred_lgbm.sum() - y_test.sum():.2f} 억원 
''')

print(f'''
실제 매출액              : {y_test.sum():.2f} 억원
RandomForest 예측 매출액 : {y_pred_rf.sum():.2f} 억원
XGBoost 예측 매출액      : {y_pred_xgb.sum():.2f} 억원
LightGBM 예측 매출액     : {y_pred_lgbm.sum():.2f} 억원
''')

print(f'''
RandomForest 예측 오차율 : {100*(y_pred_rf.sum() - y_test.sum())/y_test.sum():.2f} %
XGBoost 예측 오차율      : {100*(y_pred_xgb.sum() - y_test.sum())/y_test.sum():.2f} %
LightGBM 예측 오차율     : {100*(y_pred_lgbm.sum() - y_test.sum())/y_test.sum():.2f} %
''')

# 5-2. Model Visualization
plt.rc('figure', figsize=(12,6))

plt.plot(x_range, y_test, 
         marker='o', markersize=5, label='Actual', color='black')
plt.plot(x_range, y_pred_rf, 
         marker='o', markersize=5, label=f'RandomForest MSE: {mean_squared_error(y_pred_rf, y_test):.2f}', 
         linestyle='--', color='red')
plt.plot(x_range, y_pred_xgb,
            marker='o', markersize=5, label=f'XGBoost MSE: {mean_squared_error(y_pred_xgb, y_test):.2f}', 
            linestyle='--', color='blue')
plt.plot(x_range, y_pred_lgbm,
            marker='o', markersize=5, label=f'LightGBM MSE: {mean_squared_error(y_pred_lgbm, y_test):.2f}', 
            linestyle='--', color='green')

plt.title('Personal Sales Prediction (2024.01)')
plt.xlabel('Date')
plt.ylabel('Personal Sales (억 원)')
plt.xticks(x_range, df_test[test_period]['date'].dt.day)
plt.legend()
plt.show()

# 5-3. Model Interpretation
best_model = model_lgbm

feature_name_ko = {
    '승인일자': '승인일자',
    'year':     '연도',
    'month':    '월',
    'day':      '일',
    'dayofweek':'요일',
}

df_fi = pd.DataFrame({
    'feature':   [feature_name_ko[f] for f in X_train.columns],
    'importance': best_model.feature_importances_,
})

plt.figure(figsize=(12,6))

sns.barplot(data=df_fi, x='importance', y='feature', hue='feature', palette='crest', legend=False)
plt.title('피처 중요도 (LightGBM)')
plt.xlabel('중요도')
plt.ylabel('피처')
plt.show()


# %% References
import sklearn as sk
import xgboost as xgb
import lightgbm as lgb

print(f'numpy version: {np.__version__}')
print(f'pandas version: {pd.__version__}')
print(f'seaborn version: {sns.__version__}')
print(f'matplotlib version: {mpl.__version__}')
print(f'scikit-learn version: {sk.__version__}')
print(f'xgboost version: {xgb.__version__}')
print(f'lightgbm version: {lgb.__version__}')

# %%
