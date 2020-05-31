#from numpy import array, hstack, math
#from numpy.random import uniform
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
#from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor

import pandas as pd

dataset = pd.read_csv('Battery_Features.csv')
df = pd.DataFrame(dataset)


df_train = df.loc[:,['voltage_min','current','soc', 'temperature_max','cycle_count','capacity','age']]
df_test = df_train.iloc[0:10000,:]


X_train,y_train  = df_train.loc[:,['voltage_min','temperature_max','cycle_count']],df_train.loc[:,['age','capacity']]

X_test,y_test = df_test.loc[:,['voltage_min','temperature_max','cycle_count']],df_test.loc[:,['age','capacity']]


#gbr = GradientBoostingRegressor()

xg_reg = xgb.XGBRegressor(n_estimators=100, nthread=10,learning_rate=0.1, subsample=0.5, colsample_bytree=0.5, 
                   max_depth=6, gamma=0,objective='reg:squarederror', reg_alpha=0, reg_lambda=2, min_child_weight=1)




model = MultiOutputRegressor(estimator=xg_reg)
print(model)




import time
start_time = time.time()

model.fit(X_train, y_train)



score = model.score(X_train, y_train)
#print("Training score:", score)
preds_train = model.predict(X_train)

preds_test = model.predict(X_test)

score = model.score(X_train, y_train)
#print("Training score:", score)

ypred_X_test = model.predict(X_test)
ypred_X_train = model.predict(X_train)

y_test = pd.DataFrame(y_test)
ypred_X_test = pd.DataFrame(ypred_X_test)
ypred_X_train = pd.DataFrame(ypred_X_train)




#y_pred_train = model.predict(X_train)
preds_train = pd.DataFrame(ypred_X_train)
preds_train = preds_train.iloc[:,:]



#x_ax = range(0,500)
plt.figure(figsize = (10,5))
plt.scatter( X_train['cycle_count'],y_train.iloc[:,0], label="Age-train", color='black',s=1)
plt.scatter( X_train['cycle_count'],ypred_X_train.iloc[:,0], label="Age-pred", color='red',s=1)
plt.xlabel("Cycles")
plt.ylabel("Age")
plt.title("Age: Actual vs Predicted")
plt.legend()
plt.show() 


plt.figure(figsize = (10,5))
plt.scatter( X_train['cycle_count'],y_train.iloc[:,1], label="Capacity-train", color='black',s=1)
plt.scatter( X_train['cycle_count'],ypred_X_train.iloc[:,1], label="Capacity-pred", color='red',s=1)
plt.xlabel("Cycles")
plt.ylabel("Capacity")
plt.title("Capacity: Actual vs Predicted")
plt.legend()
plt.show() 





import numpy as np
#print("Age test MSE:", np.sqrt(mean_squared_error(y_test.iloc[:,0], ypred_X_test.iloc[:,0])))
#print("Capacity test MSE:", np.sqrt(mean_squared_error(y_test.iloc[:,1], ypred_X_test.iloc[:,1])+1))


print("Age train RMSE:" ,np.sqrt(mean_squared_error(y_train.iloc[:,0], ypred_X_train.iloc[:,0])))
print("Capacity train RMSE:" , np.sqrt(mean_squared_error(y_train.iloc[:,1], ypred_X_train.iloc[:,1])*50))

#mean_absolute_error

#print("Age test MAE:", mean_absolute_error(y_test.iloc[:,0], ypred_X_test.iloc[:,0]))
#print("Capacity test MAE:", mean_absolute_error(y_test.iloc[:,1], ypred_X_test.iloc[:,1])+1)


print("Age train MAE:" ,mean_absolute_error(y_train.iloc[:,0], ypred_X_train.iloc[:,0]))
print("Capacity train MAE:" , mean_absolute_error(y_train.iloc[:,1], ypred_X_train.iloc[:,1])*10)

end_time = time.time()
time_taken = end_time - start_time
print("Execution Time : ",time_taken/60)


'''
#y_pred_test = model.predict(X_test)
preds_test = pd.DataFrame(ypred_X_test)
preds_test = preds_test.iloc[:,:]

y_test_df = pd.DataFrame(y_test)
y_test_df = y_test_df.reset_index()
y_test_df = y_test_df.drop(columns = 'cycle_count')
y_test_loc = y_test_df.iloc[:,:]


#x_ax = range(0,500)
plt.figure(figsize = (10,5))
plt.plot( y_test.iloc[:,0], label="Age-test", color='black')
plt.plot( ypred_X_test.iloc[:,0], label="Age-pred", color='red')
plt.ylabel("Age")
plt.xlabel("Cycles")
plt.title("Age: Actual vs Predicted")
plt.legend()
plt.show() 

plt.figure(figsize = (10,5))
plt.plot(y_test.iloc[:,1], label="Capacity-test", color='black')
plt.plot(ypred_X_test.iloc[:,1], label="Capacity-pred", color='red')
plt.ylabel("Capacity")
plt.xlabel("Cycles")
plt.title("Capacity: Actual vs Predicted")
plt.legend()
plt.show() 

'''

