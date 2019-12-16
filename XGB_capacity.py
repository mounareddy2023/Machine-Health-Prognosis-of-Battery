import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

import os
os.chdir('C:\\Users\\mouna\\OneDrive\\Documents\\Battery_project\\Datasets\\temp')

dataset_25d = pd.read_csv('ScopeData_25d_cycle.csv')
df_25d = pd.DataFrame(dataset_25d)
print(df_25d.head())
print(df_25d.columns)

print(df_25d.describe())
des = df_25d.describe()
des.to_csv("C:\\Users\\mouna\\OneDrive\\Documents\\Battery_project\\Datasets\\temp\\describe_cycle.csv")
print(df_25d.corr())
corr = df_25d.corr()
corr = pd.DataFrame(corr)
#plt.figure(figsize=(10, 10))
#sb.heatmap(corr.corr(), annot=True)

import xgboost as xgb
from sklearn.metrics import mean_squared_error

X_25d = df_25d.loc[:,['voltage', 'current','soc','temperature','cycle_count']]
y_25d = df_25d.loc[:,['capacity']]
from sklearn.model_selection import train_test_split
X_train_25d, X_test_25d, y_train_25d, y_test_25d = train_test_split(X_25d, y_25d, test_size=0.2)


xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', learning_rate = 0.1, gamma=10, min_child_weight=10,
                          subsample = 0.7 ,  colsample_bytree = 0.7 ,eta = 0.1,
                max_depth = 6, alpha = 10, nthread=40, n_estimators = 100,seed = 1234)
import time
start_time = time.time()
xg_reg.fit(X_train_25d,y_train_25d)
preds_train_25d = xg_reg.predict(X_train_25d)
end_time = time.time()
time_taken = end_time - start_time
print("Execution Time : ",time_taken/60)
mse=mean_squared_error(y_train_25d, preds_train_25d)
print("MSE (train): %f" % (mse))
rmse = np.sqrt(mean_squared_error(y_train_25d, preds_train_25d))
print("RMSE (train): %f" % (rmse))
from sklearn.metrics import explained_variance_score
print('Accuracy (train):',np.abs(explained_variance_score(preds_train_25d,y_train_25d)*100))
print("objective ='reg:squarederror', gamma=10, min_child_weight=10,subsample = 0.7 ,colsample_bytree = 0.7 ,eta = 0.1")
print("max_depth = 6, learning_rate = 0.1, alpha = 10, nthread=40, n_estimators = 1000,seed = 1234")

#preds_test_25d = best_svr.predict(X_test_25d)

preds_test_25d = xg_reg.predict(X_test_25d)
mse=mean_squared_error(y_test_25d, preds_test_25d)
print("MSE (test): %f" % (mse))
rmse = np.sqrt(mean_squared_error(y_test_25d, preds_test_25d))
print("RMSE (test): %f" % (rmse))
print('Accuracy (test):',np.abs(explained_variance_score(preds_test_25d,y_test_25d)*100))

preds_test_25d_df = pd.DataFrame(preds_test_25d)
preds_test_25d_df_loc = preds_test_25d_df.iloc[10000:10100,:]

y_test_25d_df = pd.DataFrame(y_test_25d)
y_test_25d_df = y_test_25d_df.reset_index()
y_test_25d_df = y_test_25d_df.drop(columns = 'index')
y_test_25d_df_loc = y_test_25d_df.iloc[10000:10100,:]

fig, ax = plt.subplots(1, figsize=(12, 8))
ax.plot(preds_test_25d_df_loc, color='red', label='Battery capacity predicted')
plt.plot(y_test_25d_df_loc, color='black', linewidth= 1, label='Battery capacity actual')
plt.ylabel("capacity")
plt.xlabel("Time in seconds")
plt.legend()
plt.grid()
plt.title("Actual vs predicted")
plt.show()
'''
plt.rcParams['figure.figsize'] = [50, 50]
xgb.plot_tree(xg_reg,num_trees=50)
plt.show()
'''
plt.rcParams['figure.figsize'] = [5, 5]
xgb.plot_importance(xg_reg)
plt.show()

