#from numpy import array, hstack, math
#from numpy.random import uniform
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
#from sklearn.ensemble import GradientBoostingRegressor
#import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

import pandas as pd

dataset = pd.read_csv('BatteryModelling.csv')
#dataset = pd.read_csv('Battery_Features.csv')
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
df_test = df_train.iloc[1000:1100,:]


X_train,y_train  = df_train.loc[:,['voltage_min','cycle_count','soc', 'temperature_max']],df_train.loc[:,['age','capacity']]

X_test,y_test = df_test.loc[:,['voltage_min','cycle_count','soc', 'temperature_max']],df_test.loc[:,['age','capacity']]

SVR_model = SVR()




model = MultiOutputRegressor(estimator=SVR_model)
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

import numpy as np



#mean_absolute_error
'''
print("")

print("Age test RMSE:", np.sqrt(mean_squared_error(y_test.iloc[:,0], ypred_X_test.iloc[:,0])))
print("Capacity test RMSE:", np.sqrt(mean_squared_error(y_test.iloc[:,1], ypred_X_test.iloc[:,1])*1000))


print("Age test MAE:", mean_absolute_error(y_test.iloc[:,0], ypred_X_test.iloc[:,0]))
print("Capacity test MAE:", mean_absolute_error(y_test.iloc[:,1], ypred_X_test.iloc[:,1])+1)


print("Age train MAE:" ,mean_absolute_error(y_train.iloc[:,0], ypred_X_train.iloc[:,0]))
print("Capacity train MAE:" , mean_absolute_error(y_train.iloc[:,1], ypred_X_train.iloc[:,1])+1)
'''


#y_pred_train = model.predict(X_train)
preds = pd.DataFrame(ypred_X_train)
preds = preds.iloc[:,:]



#x_ax = range(0,500)
plt.figure(figsize = (10,5))
plt.scatter( X_train.loc[:,['cycle_count']],y_train.iloc[:,0], label="Age-train", color='black',s=0.1)
plt.scatter( X_train.loc[:,['cycle_count']],preds.iloc[:,0], label="Age-pred", color='red',s=0.1)
plt.ylabel("Age")
plt.xlabel("Cycle")
plt.title("Age: Actual vs Predicted")
plt.legend()
plt.show() 

plt.figure(figsize = (10,5))
plt.scatter(  X_train.loc[0:1000,['cycle_count']],y_train.iloc[0:1000,1], label="Capacity-train", color='black',s=0.1)
plt.scatter(  X_train.loc[0:1000,['cycle_count']],preds.iloc[0:1000,1], label="Capacity-pred", color='red',s=0.1)
plt.ylabel("Capacity")
plt.xlabel("Cycle")
plt.title("Capacity: Actual vs Predicted")
plt.legend()
plt.show() 


print("")

print("Age train RMSE:" ,np.sqrt(mean_squared_error(y_train.iloc[:,0], ypred_X_train.iloc[:,0])))
print("Capacity train RMSE:" , np.sqrt(mean_squared_error(y_train.iloc[:,1], ypred_X_train.iloc[:,1])*1000))


end_time = time.time()
time_taken = end_time - start_time
print("")
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
plt.scatter( X_test['cycle_count'], y_test.iloc[:,0], label="Age-test", color='black',s=5)
plt.scatter( X_test['cycle_count'], ypred_X_test.iloc[:,0], label="Age-pred", color='red',s=5)
plt.ylabel("Age")
plt.xlabel("Cycles")
plt.title("Age: Actual vs Predicted")
plt.legend()
plt.show() 

plt.figure(figsize = (10,5))
plt.scatter( X_test['cycle_count'],y_test.iloc[:,1], label="Capacity-test", color='black',s=5)
plt.scatter( X_test['cycle_count'],ypred_X_test.iloc[:,1], label="Capacity-pred", color='red',s=5)
plt.ylabel("Capacity")
plt.xlabel("Cycles")
plt.title("Capacity: Actual vs Predicted")
plt.legend()
plt.show() 


'''

'''
test = pd.DataFrame(columns = ['voltage_min','cycle_count','soc', 'temperature_max'])
test= test.append({'voltage_min':3.0016,'cycle_count':2800,'soc':20, 'temperature_max':48.6},ignore_index=True)

y_pred_test = model.predict(test)
preds_test = pd.DataFrame(y_pred_test)
preds_test = preds_test.iloc[:,:]

y_test_df = pd.DataFrame(y_test)
y_test_df = y_test_df.reset_index()
y_test_df = y_test_df.drop(columns = 'cycle_count')
y_test_loc = y_test_df.iloc[:,:]


#x_ax = range(0,500)
plt.figure(figsize = (10,5))
plt.scatter(X_train.cycle_count, y_train.iloc[:,0], label="Age-test", color='black',s=1)
plt.scatter(test.cycle_count, preds_test.iloc[:,0], label="Age-pred", color='red',s=10)
plt.ylabel("Age")
plt.xlabel("Cycles")
plt.title("Age: Actual vs Predicted")
plt.legend()
plt.show() 


#x_ax = range(0,500)
plt.figure(figsize = (10,5))
plt.scatter(X_train.cycle_count, y_train.iloc[:,1], label="Capacity-test", color='black',s=1)
plt.scatter(test.cycle_count, preds_test.iloc[:,1], label="Capacity-pred", color='red',s=10)
plt.ylabel("Capacity")
plt.xlabel("Cycles")
plt.title("Capacity: Actual vs Predicted")
plt.legend()
plt.show() 


'''







