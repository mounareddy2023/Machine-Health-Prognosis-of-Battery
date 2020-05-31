#from numpy import array, hstack, math
#from numpy.random import uniform
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
#from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
import pandas as pd

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
df = data.loc[:,['voltage_min','soc', 'temperature_max','cycle_count','capacity','age']]
#df_test = df.iloc[0:1000,:]
import pandas as pd 

from sklearn.model_selection import  cross_val_score, LeaveOneOut  


from sklearn.model_selection import  cross_val_predict
from sklearn import metrics

X = df.loc[:,['voltage_min','cycle_count','soc', 'temperature_max']]
y= df.loc[:,['age','capacity']]

#X_train, X_test, y_train, y_test = train_test_split(X, y)


xg_regr = xgb.XGBRegressor()

xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.3,
                 max_depth = 9,gamma=0, alpha = 10, n_estimators = 100)


SVR_model = SVR()


model = MultiOutputRegressor(estimator=xg_reg)
print(model)


#(n_estimators = 100,learning_rate=0.1, subsample=0.5, colsample_bytree=0.5, 
#            max_depth=6, gamma=0, reg_alpha=0, reg_lambda=2, min_child_weight=1)

loo = LeaveOneOut()
loo.get_n_splits(X)


for train_index, test_index in loo.split(X):
   #print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X.iloc[train_index], X.iloc[test_index]
   y_train, y_test = y.iloc[train_index], y.iloc[test_index]
   #print(X_train, X_test, y_train, y_test)

# Perform 6-fold cross validation
scores = cross_val_score(model, X, y, cv=6)
print('Cross-validated scores:', scores)
predictions = cross_val_predict(model, X, y, cv=6)


preds = pd.DataFrame(predictions)
preds = preds.iloc[:,:]

accuracy_age = metrics.r2_score(y.iloc[:,0], preds.iloc[:,0])

accuracy_capacity = metrics.r2_score(y.iloc[:,1], preds.iloc[:,1])



#x_ax = range(0,500)
plt.figure(figsize = (10,5))
plt.plot( X['cycle_count'],y.iloc[:,0], label="Age-train", color='black')
plt.plot( X['cycle_count'],preds.iloc[:,0], label="Age-pred", color='red')
plt.ylabel("Age")
plt.xlabel("Cycle")
plt.title("Age: Actual vs Predicted")
plt.legend()
plt.show() 

plt.figure(figsize = (10,5))
plt.plot( X['cycle_count'],y.iloc[:,1], label="Capacity-train", color='black')
plt.plot( X['cycle_count'],preds.iloc[:,1], label="Capacity-pred", color='red')
plt.ylabel("Capacity")
plt.xlabel("Cycle")
plt.title("Capacity: Actual vs Predicted")
plt.legend()
plt.show() 



print('Cross-Predicted Accuracy Age :', accuracy_age*100)
print('Cross-Predicted Accuracy Capacity :', accuracy_capacity*100)















