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

from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut  


from sklearn.model_selection import  cross_val_predict
from sklearn import metrics

X = df.loc[:,['voltage_min','soc', 'temperature_max','cycle_count']]
y= df.loc[:,['age']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


#model = xgb.XGBRegressor()


model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.3, subsample=0.9, colsample_bytree=1, 
                   max_depth=9, gamma=0, reg_alpha=0, reg_lambda=2, min_child_weight=7)

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
scores = cross_val_score(model, X, y, cv=7)
print('Cross-validated scores:', scores)
predictions = cross_val_predict(model, X, y, cv=7)
accuracy = metrics.r2_score(y, predictions)
print('Cross-Predicted Accuracy:', accuracy)
#from sklearn.model_selection import cross_val_score

fig, ax = plt.subplots(1, figsize=(8, 5))
plt.plot(predictions, color='red', label='Battery age predicted')
plt.plot(y, color='black', linewidth= 1, label='Battery age actual')
plt.ylabel("age")
plt.xlabel("cycles")
plt.legend()
plt.title("Actual vs predicted")
plt.show()
