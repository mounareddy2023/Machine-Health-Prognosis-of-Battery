import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import xgboost as xgb
from sklearn.metrics import mean_squared_error,mean_absolute_error

#import os
#os.chdir('C:\\Users\\mouna\\OneDrive\\Documents\\Battery_project\\Datasets\\temp')

dataset = pd.read_csv('BatteryModelling.csv')
df = pd.DataFrame(dataset)
datasetdf = pd.DataFrame(dataset)
Mean = datasetdf.groupby(['cycle_count'])['voltage','current','soc', 'temperature','capacity','age','cycle_count'].mean()
Min=datasetdf.groupby(['cycle_count'])['voltage','soc', 'temperature'].min()
Min.rename(columns = {'voltage':'voltage_min', 'soc':'soc_min', 
                              'temperature':'temperature_min'}, inplace = True)
Max=datasetdf.groupby(['cycle_count'])['voltage','soc', 'temperature'].max()
Max.rename(columns = {'voltage':'voltage_max', 'soc':'soc_max', 
                              'temperature':'temperature_max'}, inplace = True)
#count=datasetdf.groupby(['cycle_count'])['cycle_count'].count()

datadf = pd.concat([Mean,Min,Max], axis=1)

data = datadf[['voltage','voltage_min','voltage_max','current','soc','soc_min','soc_max',
         'temperature','temperature_min','temperature_max','cycle_count','capacity','age']]

data1 = datadf[['voltage','voltage_min','voltage_max','current','soc','soc_min',
         'temperature','temperature_min','temperature_max','cycle_count','capacity','age']]

#df_train = pd.read_csv('file:///C:/Users/mouna/Documents/Battery_project/Datasets/temp/Lithium_discharge.csv')
df_train = data.loc[:,['voltage_min','cycle_count','soc', 'temperature_max','capacity','age']]
df_test = df_train.iloc[1000:2100,:]

X, y = data.loc[:,['voltage_min','cycle_count','soc', 'temperature_max']],data.loc[:,['age']]

X_train,y_train = X.loc[:,['voltage_min','cycle_count','soc', 'temperature_max']],y.loc[:,['age']]

X_test,y_test = X.loc[:,['voltage_min','cycle_count','soc', 'temperature_max']],y.loc[:,['age']]


data_dmatrix = xgb.DMatrix(data=X,label=y)


xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 1, learning_rate = 0.3,
                max_depth = 9, alpha = 10, n_estimators = 1000)


xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))



params = {"objective":"reg:linear",'colsample_bytree': 1,'learning_rate': 0.1,
                'max_depth': 9, 'alpha': 10, 'n_estimators' : 10}

cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=100, early_stopping_rounds=10,
                    metrics="rmse", as_pandas=True, seed=1)

plt.figure(figsize=(10,5))
plt.plot(cv_results['test-rmse-mean'],label='test-rmse-mean')
plt.plot(cv_results['train-rmse-mean'],label='train-rmse-mean')
plt.xlabel("num_boost_round")
plt.ylabel("rmse")
plt.title("rmse")
plt.legend()
plt.show()




print(cv_results.head())
print(cv_results.tail())

print((cv_results["test-rmse-mean"]).tail(1))
print((cv_results["train-rmse-mean"]).tail(1))


'''

y_pred_train = xg_reg.predict(X_train)
preds_train = pd.DataFrame(y_pred_train)
preds_train = preds_train.iloc[:,:]


fig, ax = plt.subplots(1, figsize=(10, 5))
ax.plot(preds_train, color='red', label='Battery age predicted')
ax.plot(y_train, color='black', linewidth= 1, label='Battery age actual')
plt.ylabel("age")
plt.xlabel("Time in seconds")
plt.legend()
plt.title("Actual vs predicted")
plt.show()

'''
'''
xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
num_boost_round=100



cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    seed=42,
    nfold=5,
    metrics={'mae'},
    early_stopping_rounds=10
)


cv_results
plt.figure(figsize=(5,5))
plt.plot(cv_results['test-mae-mean'])
plt.plot(cv_results['train-mae-mean'])
cv_results['test-mae-mean'].min()
cv_results['train-mae-mean'].min()

'''
'''
gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(9,12)
    for min_child_weight in range(5,8)
]

min_mae = float("Inf")
best_params = None
for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(
                             max_depth,
                             min_child_weight))    # Update our parameters
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=10
    )    # Update best MAE
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].argmin()
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (max_depth,min_child_weight)


print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))
'''
'''
params['max_depth'] = 9
params['min_child_weight'] = 7
gridsearch_params = [
    (subsample, colsample)
    for subsample in [i/10. for i in range(7,11)]
    for colsample in [i/10. for i in range(7,11)]
]
min_mae = float("Inf")
best_params = None# We start by the largest values and go down to the smallest
for subsample, colsample in reversed(gridsearch_params):
    print("CV with subsample={}, colsample={}".format(
                             subsample,
                             colsample))    # We update our parameters
    params['subsample'] = 0.9
    params['colsample_bytree'] = 1.0    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=10
    )    # Update best score
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].argmin()
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (subsample,colsample)
        
print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))
#print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))



'''




'''
# This can take some time…
min_mae = float("Inf")
best_params = None
for eta in [.3, .2, .1, .05, .01, .005]:
    print("CV with eta={}".format(eta))    # We update our parameters
    params['eta'] = eta    # Run and time CV
    cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=5,
            metrics=['mae'],
            early_stopping_rounds=10)    # Update best score
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].argmin()
    print("\tMAE {} for {} rounds\n".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = eta
        
print("Best params: {}, MAE: {}".format(best_params, min_mae))
        
        
        
'''
'''
xgb.plot_tree(xg_reg,num_trees=0)
#plt.rcParams['figure.figsize'] = [50, 10]
plt.show()



xgb.plot_importance(xg_reg)
#plt.rcParams['figure.figsize'] = [5, 5]
#plt.show()

'''


