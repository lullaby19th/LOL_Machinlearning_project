import xgboost as xgb
from xgboost import plot_importance
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import plot_importance, plot_tree
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

regular = pd.read_excel("d:\\data\\lol\\lol_final_dataset.xlsx")
# print(regular.columns)
# print(regular.info()) 
regular['MVP'] = regular['MVP'] / 100
regular['MVP'].fillna(0, inplace = True) # 결측치 처리


lol = regular.sort_values(by = ['시즌','아이디'], ascending = True)
lol.drop(['Unnamed: 0'], axis =1 , inplace = True) # 불필요한 컬럼 제거
print(lol.head())
print(lol.columns)

#%%
# lol.to_excel("d:\\data\\lol\\sort_name_lol.xlsx")

list1 = list(lol['아이디'].unique())
print(list1)

lol['생년월일'] = lol['생년월일'].astype(str)
lol['출생년도'] = lol['생년월일'].str.extract('([0-9]+)', expand = False)

print(lol['출생년도'])

print(lol['출생년도'].isnull().sum())
print('\n\n')

lol.drop(['생년월일'], axis =1 , inplace = True)
# print(lol[:][lol['아이디'] == 'Bdd'])


for i in list1:
    
    most_freq = lol[:][lol['아이디'] == i]['출생년도'].value_counts(dropna=True).idxmax()
    # print(type(most_freq))
    freq1 = int(most_freq)
    
    
    lol.loc[lol['아이디'] == i, '출생년도'] = freq1
    
    #print(lol[:][lol['아이디'] == i])
    
print(lol['출생년도'].isnull().sum())
print(lol['출생년도'].value_counts())
print('\n\n')

lol['나이'] = 2020 - lol['출생년도']
lol.drop(['출생년도'], axis=1, inplace=True)
# lol = lol[:][lol['출전횟수'] >= 5] 
#%%
# 분당와드클리어 결측치 처리하기
# mean_ward = lol[:][lol['아이디'] == 'Bdd']['분당와드클리어'].mean(axis=0)
# print(mean_ward)

print(lol['분당와드클리어'].isnull().sum())
print('\n\n')    


for i in list1:
    mask1 = lol['아이디'] == i
    mean_ward = lol[:][lol['아이디'] == i]['분당와드클리어'].mean(axis=0)
    print(i, mean_ward)
    
    lol.loc[mask1,'분당와드클리어'] = lol.loc[mask1,'분당와드클리어'].fillna(mean_ward)

lol.dropna(subset = ['분당와드클리어'], axis = 0, inplace = True) # NaN =누락데이터를 제거함.

print(lol['분당와드클리어'].isnull().sum())
print('\n\n')    
print(lol[:][lol['아이디']=='ADD']['분당와드클리어'])
print(lol['분당와드클리어'].value_counts())
print('\n\n')    

# lol.to_excel("d:\\data\\lol\\sort_name_lol_sb.xlsx")

#%%
# '15CS비율' 결측치 처리하기
# mean_ward = lol[:][lol['아이디'] == 'Bdd']['분당와드클리어'].mean(axis=0)
# print(mean_ward)

print(lol['15CS비율'].isnull().sum())
print('\n\n')    


for i in list1:
    mask1 = lol['아이디'] == i
    mean_ward = lol[:][lol['아이디'] == i]['15CS비율'].mean(axis=0)
    print(i, mean_ward)
    
    lol.loc[mask1,'15CS비율'] = lol.loc[mask1,'15CS비율'].fillna(mean_ward)

lol.dropna(subset = ['15CS비율'], axis = 0, inplace = True) # NaN =누락데이터를 제거함.

print(lol['15CS비율'].isnull().sum())
print('\n\n')    
print(lol[:][lol['아이디']=='ADD']['15CS비율'])
print(lol['15CS비율'].value_counts())
print('\n\n')    

# lol.to_excel("d:\\data\\lol\\sort_name_lol_sb.xlsx")
#%%


local_std1 = lol['GD10'].std() * 5

# 이상치 값들이 무엇이 있는지 확인!
result1 = lol['GD10'][lol['GD10'] > local_std1]
print(result1)
print(lol.shape)
#%%
lol = lol[:][lol['GD10'] < local_std1]
print(lol.shape)
#%%
local_std2 = lol['분당CS'].std() * 4

# 이상치 값들이 무엇이 있는지 확인!
result1 = lol['분당CS'][lol['분당CS'] > local_std2]
print(result1)
print(lol.shape)

#%%
lol = lol[:][lol['분당CS'] < local_std2]
print(lol.shape)

#%%
# 킬과 출전횟수가 전체평균 75%이상인 파생변수
mask_r = (lol.K>=101) | (lol.출전횟수>=44) 
lol['K_출전횟수'] = mask_r.astype(int)
#%%
# 파생변수 추가(매핑추가)
lol.loc[(lol['K'] <= 19) & (lol['승률'] <= 0.31)  & (lol['출전횟수'] > 5), 'K승률'] = 1,
lol.loc[(lol['K'] > 19 ) & (lol['K']  <= 52) & (lol['승률'] > 0.31) & (lol['승률'] <= 0.50) & (lol['출전횟수'] > 5), 'K승률'] = 2,
lol.loc[(lol['K'] > 52 ) & (lol['K']  <= 101) & (lol['승률'] > 0.50) & (lol['승률'] <= 0.65) & (lol['출전횟수'] > 5) , 'K승률'] = 3,
lol.loc[(lol['K'] > 101 ) & (lol['K']  <= 208) & (lol['승률'] > 0.65) & (lol['승률'] <= 1) & (lol['출전횟수'] > 5), 'K승률'] = 4
lol['K승률'].fillna (0, inplace=True)

#%%
#  MVP 매핑

print(lol.MVP.describe())
count, bin_dividers = np.histogram(lol['MVP'], bins = 4)
    
print(count) 
print(bin_dividers)   


#%%    
title_mapping = {0:4, 1:4, 2:4,
                3:4, 4:3, 5:3, 6:3, 7: 3, 8: 2, 9: 2, 10: 2, 11: 1,
                 12: 1, 13: 1, 14:1} # 이름 성 마다 숫자지정


lol['Tier'] = lol['MVP'].map(title_mapping) 
lol.drop(['MVP'], axis=1, inplace=True)

#%%
# 파생변수 추가 (나이가 늙으면은 은퇴한다.)

lol.loc[(lol['나이'] <= 20) & (lol['출전횟수'] > 5), '은퇴'] = 4,
lol.loc[(lol['나이'] > 20) & (lol['나이'] <= 23)  & (lol['출전횟수'] > 5), '은퇴'] = 3,
lol.loc[(lol['나이'] > 23) & (lol['나이'] <= 25)  & (lol['출전횟수'] > 5), '은퇴'] = 2,
lol.loc[(lol['나이'] > 25) & (lol['나이'] <= 40)  & (lol['출전횟수'] > 5), '은퇴'] = 1,
lol['은퇴'].fillna(0, inplace=True)

#print(lol['은퇴'].value_counts())
#%% 
# 파생변수 추가

lol.loc[(lol['K'] <= 8) & (lol['KDA'] <= 2.2 ) & (lol['출전횟수'] > 5), 'KA'] = 4,
lol.loc[(lol['K'] > 8) & (lol['K'] <= 31) & (lol['KDA'] > 2.2) & (lol['KDA'] <= 3.3) & (lol['출전횟수'] > 5), 'KA'] = 3,
lol.loc[(lol['K'] > 31) & (lol['K'] <= 89) & (lol['KDA'] > 3.3) & (lol['KDA'] <= 4.4)  & (lol['출전횟수'] > 5), 'KA'] = 2,
lol.loc[(lol['K'] > 89) & (lol['K'] <= 207) & (lol['KDA'] > 4.4) & (lol['KDA'] <= 18.5)  & (lol['출전횟수'] > 5), 'KA'] = 1,
lol['KA'].fillna(method='ffill', inplace=True)
#%%
# train, test 나누기
train = lol.loc[lol['경기년도'] <= 2018]
test = lol.loc[lol['경기년도'] == 2019]


X1 = train[[ '출전횟수', '승률', 'K',  'KDA', '킬관여울', '데스관여율',
       '퍼스트블러드',  '분당CS', '15CS비율', 'DPM', '골드비율', 'DMG',
        '경기년도', '나이', 'K승률', '은퇴', 'CS10', 'KA', '시즌', '분당와드', 'GD10']]

X2 = test[[ '출전횟수', '승률', 'K', 'KDA', '킬관여울', '데스관여율',
       '퍼스트블러드',  '분당CS', '15CS비율', 'DPM', '골드비율', 'DMG',
         '경기년도', '나이', 'K승률', '은퇴', 'CS10', 'KA', '시즌', '분당와드', 'GD10']]

y = train['Tier']
y_test = test['Tier']

# train, test 데이터 shape 확인하기!
print('train data 개수: ', X1.shape)
print('test data 개수: ', X2.shape)
print('\n')

from sklearn import preprocessing
X1 = preprocessing.StandardScaler().fit(X1).transform(X1) # train 데이터 정규화
X2 = preprocessing.StandardScaler().fit(X2).transform(X2) # test 데이터 정규화

print(X1)
print(X2)



#%%
# 머신러닝 기법 코드
from xgboost import XGBClassifier
from sklearn import metrics

xgb_model = XGBClassifier()
xgb_model.fit(X1,y)


# 7단계. 테스트 데이터로 예측하기
y_pred = xgb_model.predict(X2)

print(y_pred[0:20]) # 예측데이터 10개
print(y_test.values[0:20]) # 실제 데이터 10개





# F1-report 확인코드
from sklearn import metrics

f1_report = metrics.classification_report(y_test, y_pred)
print(f1_report,'\n')

# 9단계 정확도 확인
from sklearn.metrics import accuracy_score
accuracy = round(accuracy_score(y_test, y_pred),4)

print('정확도 = ' , accuracy) # 1.0


# MSE 구하기 (mean square error)
real = test['Tier']
pred = y_pred
# ab = test['출전횟수']


from sklearn.metrics import mean_squared_error
mse = round(mean_squared_error(real,pred) ** 0.5,4)
print('mse = ', mse)

#%%
hr2_accuracy_list = []

max_depth_list = [1,3,5,7,9,11,13]
subsample_list = [0.2, 0.4, 0.6, 0.8, 1.0]

def hr2_grid_search(max_depth, subsample): 
    xgb_model = xgb.XGBClassifier(max_depth=max_depth,
                              subsample=subsample)
    xgb_pred = xgb_model.fit(X1, y).predict(X2)
    
    return([max_depth, subsample, accuracy_score(y_test, xgb_pred)])
      

for max_depth in max_depth_list:
    for subsample in subsample_list:
        hr2_accuracy_list.append(hr2_grid_search(max_depth, subsample))
    
hr2_xgb_df = pd.DataFrame(hr2_accuracy_list, columns=['tree depth', 'subsample rate', 'accuracy'])

result = hr2_xgb_df.sort_values(by = 'accuracy', ascending = False)
print(result.iloc[0])
print('max_depth', result.iloc[0,0])
print('max_subsample_rate', result.iloc[0,1])



#%%
xgb_model = XGBClassifier(max_depth= result.iloc[0,0],
                              subsample = result.iloc[0,1])

print(xgb_model)
#%%
xgb_model = XGBClassifier(max_depth= result.iloc[0,0],
                              subsample = result.iloc[0,1], seed=0)


xgb_model.fit(X1,y)


# 7단계. 테스트 데이터로 예측하기
y_pred = xgb_model.predict(X2)

print(y_pred[0:20]) # 예측데이터 10개
print(y_test.values[0:20]) # 실제 데이터 10개





# F1-report 확인코드
from sklearn import metrics

f1_report = metrics.classification_report(y_test, y_pred)
print(f1_report,'\n')

# 9단계 정확도 확인
from sklearn.metrics import accuracy_score
accuracy = round(accuracy_score(y_test, y_pred),4)

print('정확도 = ' , accuracy) # 1.0


# MSE 구하기 (mean square error)
real = test['Tier']
pred = y_pred
# ab = test['출전횟수']


from sklearn.metrics import mean_squared_error
mse = round(mean_squared_error(real,pred) ** 0.5,4)
print('mse = ', mse)

#%%
# 결과 파일 뽑기 (submission.xlsx로 저장)

test = lol.loc[lol['경기년도'] == 2019]
pred = y_pred
df = pd.DataFrame({'아이디':test['아이디'], '포지션' : test['포지션'], '출전횟수' : test['출전횟수'] , '출전횟수' : test['출전횟수'], 'Tier':pred})
df = df.sort_values(by = '출전횟수', ascending = True)

list1 = list(df['포지션'].unique())
# print(list1)
# print(df.head())

df = df[:][df['출전횟수']>3]
#%%

for i in list1:
    
    df1 = df[:][df['포지션']== i]
    
    grouped = df1.groupby(['아이디'])
    g_mean = grouped.mean()
    
    
    result = g_mean.sort_values(by = 'Tier', ascending = True)
    count, bin_dividers = np.histogram(result['Tier'], bins = 7)
    
    print(count) 
    print(bin_dividers)   

    bin_names = ['S', 'A+', 'A-', 'B+', 'B-', 'C', 'D']
    result['Tier'] = pd.cut(x = result['Tier'], bins = bin_dividers, labels = bin_names, include_lowest = True) # 첫 경계값 포함 
    result = result.sort_values(by = 'Tier' , ascending = True)
    result.drop(['출전횟수'], axis=1, inplace=True)
# =============================================================================
#     print(i)
#     print(result['Tier'].describe())
#     print('\n')
# =============================================================================
    
    data_name = "d:\\data\\lol\\lol_submission_sub_" + i + ".xlsx"
    result.to_excel(data_name,index=True)