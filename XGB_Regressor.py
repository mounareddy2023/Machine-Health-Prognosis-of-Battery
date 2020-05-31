import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

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


#df_train = pd.read_csv('file:///C:/Users/mouna/Documents/Battery_project/Datasets/temp/Lithium_discharge.csv')
df_train = data.loc[:,['voltage_min','cycle_count','soc', 'temperature_max','capacity','age']]
df_test = df_train.iloc[1000:2100,:]


X_train,y_train = df_train.loc[:,['voltage_min','cycle_count','soc', 'temperature_max']],df_train.loc[:,['age']]

X_test,y_test = df_test.loc[:,['voltage_min','cycle_count','soc', 'temperature_max']],df_test.loc[:,['age']]


#plt.figure(figsize=(10, 10))
#sb.heatmap(corr.corr(), annot=True)

import xgboost as xgb
from sklearn.metrics import mean_squared_error,mean_absolute_error


xg_reg = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.3, subsample=0.9, colsample_bytree=1, 
                   max_depth=9, gamma=0, reg_alpha=0, reg_lambda=2, min_child_weight=7)

import time
start_time = time.time()



xg_reg.fit(X_train,y_train)
#preds_train = xg_reg.predict(X_train)


end_time = time.time()
time_taken = end_time - start_time
print("Execution Time : ",time_taken/60)



y_pred_train = xg_reg.predict(X_train)
preds_train = pd.DataFrame(y_pred_train)
preds_train = preds_train.iloc[:,:]


plt.figure(figsize=(10, 5))
plt.plot(preds_train.iloc[0:100,:], color='red', label='Battery age predicted')
plt.plot(y_train.iloc[0:100,:], color='black', linewidth= 1, label='Battery age actual')
plt.ylabel("age")
plt.xlabel("Time in seconds")
plt.legend()
plt.title("Actual vs predicted")
plt.show()


mse=mean_squared_error(y_train, y_pred_train)
print("MSE (train): %f" % (mse))
rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
print("RMSE (train): %f" % (rmse))
#from sklearn.metrics import explained_variance_score
#print('Accuracy (train):',np.abs(explained_variance_score(preds_train,y_train)*100))



y_pred_test = xg_reg.predict(X_test)
preds_test = pd.DataFrame(y_pred_test)
preds_test = preds_test.iloc[:,:]

y_test_df = pd.DataFrame(y_test)
y_test_df = y_test_df.reset_index()
y_test_df = y_test_df.drop(columns = 'cycle_count')
y_test_loc = y_test_df.iloc[:,:]

plt.figure(figsize=(10, 5))
plt.plot(preds_test.iloc[0:10,:], color='red', label='Battery age predicted')
plt.plot(y_test_loc.iloc[0:10,:], color='black', linewidth= 1, label='Battery age actual')
plt.ylabel("age")
plt.xlabel("Time in seconds")
plt.legend()
plt.title("Actual vs predicted")
plt.show()


#print("objective ='reg:squarederror', gamma=10, min_child_weight=10,subsample = 0.7 ,colsample_bytree = 0.7 ,eta = 0.1")
#print("max_depth = 6, learning_rate = 0.1, alpha = 10, nthread=40, n_estimators = 10000,seed = 1234")

#preds_test_25d = best_svr.predict(X_test_25d)

#preds_test = xg_reg.predict(X_test)
mse=mean_squared_error(y_test, y_pred_test)
#print("MSE (test): %f" % (mse))
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
print("RMSE (test): %f" % (rmse))
mae= mean_absolute_error(y_test, y_pred_test)
print("MAE (test): %f" % (mae))
#print('Accuracy (test):',np.abs(explained_variance_score(y_pred_test,y_test)*100))


plt.rcParams['figure.figsize'] = [5, 5]
xgb.plot_importance(xg_reg)
plt.show()




'''
preds_test_df = pd.DataFrame(preds_test)
preds_test_df_loc = preds_test_df.iloc[10000:10100,:]

y_test_df = pd.DataFrame(y_test)
y_test_df = y_test_df.reset_index()
y_test_df = y_test_df.drop(columns = 'index')
y_test_df_loc = y_test_df.iloc[10000:10100,:]

fig, ax = plt.subplots(1, figsize=(12, 8))
ax.plot(preds_test_df_loc, color='red', label='Battery age predicted')
plt.plot(y_test_df_loc, color='black', linewidth= 1, label='Battery age actual')
plt.ylabel("age")
plt.xlabel("Time in seconds")
plt.legend()
plt.grid()
plt.title("Actual vs predicted")
plt.show()
'''



'''
plt.rcParams['figure.figsize'] = [50, 50]
xgb.plot_tree(xg_reg,num_trees=50)
plt.show()
'''

