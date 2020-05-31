# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 20:20:44 2020

@author: mouna
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.linear_model import Lasso, ElasticNet, Ridge

from xgboost import XGBRegressor


from sklearn.feature_selection import RFECV
#from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import make_scorer 
from sklearn.base import BaseEstimator, RegressorMixin

from sklearn.metrics import mean_squared_error

# neural networks
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers

# ignore Deprecation Warning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# load the data

dataset = pd.read_csv('BatteryModelling.csv')
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
df_train = data.loc[:,['voltage_min','soc','cycle_count', 'temperature_max','capacity','age']]
df_test = df_train.iloc[1000:2100,:]
df = df_train.append(df_test , ignore_index = True)

# basic inspection
df_train.shape, df_test.shape, df_train.columns.values

#Feature Selection
X,Y = df_train.loc[:,['voltage_min','soc','cycle_count', 'temperature_max']],df_train.loc[:,['age','capacity']]

X_train,y_train = df_train.loc[:,['voltage_min','soc','cycle_count', 'temperature_max']],df_train.loc[:,['age','capacity']]

X_test,y_test = df_test.loc[:,['voltage_min','soc','cycle_count', 'temperature_max']],df_test.loc[:,['age','capacity']]

'''
xgb = XGBRegressor()
xgb.fit(X_train, y_train)
imp = pd.DataFrame(xgb.feature_importances_ ,columns = ['Importance'],index = X_train.columns)
imp = imp.sort_values(['Importance'], ascending = False)

print(imp)

# Define a function to calculate RMSE
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true-y_pred)**2))

# Define a function to calculate negative RMSE (as a score)
def nrmse(y_true, y_pred):
    return -1.0*rmse(y_true, y_pred)

neg_rmse = make_scorer(nrmse)

estimator = XGBRegressor()
selector = RFECV(estimator, cv = 3, n_jobs = -1, scoring = neg_rmse)
selector = selector.fit(X_train, y_train)

print("The number of selected features is: {}".format(selector.n_features_))

features_kept = X_train.columns.values[selector.support_] 

X_train = selector.transform(X_train)  
X_test = selector.transform(X_test)

# transform it to a numpy array so later we can feed it to a neural network
y_train = y_train.values 

'''

#------------------------------------------------------------------------------------

#Modeling and Prediction
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor



xgb_reg = xgb.XGBRegressor(n_estimators=100, learning_rate=0.3, subsample=0.9, colsample_bytree=1, 
                   max_depth=9, gamma=0, reg_alpha=0, reg_lambda=2, min_child_weight=7)
#n_estimators=100, learning_rate=0.3, subsample=0.9, colsample_bytree=1, 
#                   max_depth=9, gamma=0, reg_alpha=0, reg_lambda=2, min_child_weight=7



xgb = MultiOutputRegressor(estimator=xgb_reg)
print(xgb)


nn = Sequential()

# Initialising the ANN
nn = Sequential()

# Adding the input layer and the first hidden layer
nn.add(Dense(32, activation = 'relu', input_dim = 4))

# Adding the second hidden layer
nn.add(Dense(units = 64, activation = 'relu'))

# Adding the third hidden layer
nn.add(Dense(units = 32, activation = 'relu'))

nn.add(Dense(2))


# Compile the NN
nn.compile(loss='mse', optimizer='adam')


class Ensemble(BaseEstimator, RegressorMixin):
    def __init__(self, regressors=None):
        self.regressors = regressors
        
    def level0_to_level1(self, X):
        self.predictions_ = []

        for regressor in self.regressors:
            self.predictions_.append(regressor.predict(X).reshape(X.shape[0],2))

        return np.concatenate(self.predictions_, axis=1)
    
    def fit(self, X, y):
        for regressor in self.regressors:
            if regressor != nn:
                regressor.fit(X, y)
            else: regressor.fit(X, y, batch_size=64, epochs=100) # Neural Network
            
        self.new_features = self.level0_to_level1(X)
        
        # using a large L2 regularization to prevent the ensemble from biasing toward 
        # one particular base model
        self.combine = Ridge(alpha=10, max_iter=50000)   
        self.combine.fit(self.new_features, y)

        self.coef_ = self.combine.coef_

    def predict(self, X):
        self.new_features = self.level0_to_level1(X)
            
        return self.combine.predict(self.new_features).reshape(X.shape[0],2)




import time
start_time = time.time()

#model = Ensemble(regressors=[xgb,las,ridge,elast,nn])

Ensemble_model = Ensemble(regressors=[xgb,nn])

Ensemble_model.fit(X_train, y_train)
#print("\nThe weights of the five base models are: {}".format(Ensemble_model.coef_))


'''
y_pred_train = Ensemble_model.predict(X_train)

preds_train = pd.DataFrame(y_pred_train)
preds_train = preds_train.iloc[:,:]


mse=mean_squared_error(y_train, y_pred_train)
#print("MSE (train): %f" % (mse))
rmser = np.sqrt(mean_squared_error(y_train, y_pred_train))
print("RMSE (train): %f" % (rmser))

'''



ypred_X_test = Ensemble_model.predict(X_test)
ypred_X_test = pd.DataFrame(ypred_X_test)

ypred_X_train = Ensemble_model.predict(X_train)
ypred_X_train = pd.DataFrame(ypred_X_train)

y_test = pd.DataFrame(y_test)




#y_pred_train = model.predict(X_train)
preds_train = pd.DataFrame(ypred_X_train)
preds_train = preds_train.iloc[:,:]



#x_ax = range(0,500)
plt.figure(figsize = (10,5))
plt.plot( y_train.iloc[0:100,0], label="Age-train", color='black')
plt.plot( ypred_X_train.iloc[0:100,0], label="Age-pred", color='red')
plt.xlabel("Cycles")
plt.ylabel("Age")
plt.title("Age: Actual vs Predicted")
plt.legend()
plt.show() 


plt.figure(figsize = (10,5))
plt.plot( y_train.iloc[0:100,1], label="Capacity-train", color='black')
plt.plot( ypred_X_train.iloc[0:100,1], label="Capacity-pred", color='red')
plt.xlabel("Cycles")
plt.ylabel("Capacity")
plt.title("Capacity: Actual vs Predicted")
plt.legend()
plt.show() 



print("Age train RMSE:" ,np.sqrt(mean_squared_error(y_train.iloc[:,0], ypred_X_train.iloc[:,0])))
print("Capacity train RMSE:" , np.sqrt(mean_squared_error(y_train.iloc[:,1], ypred_X_train.iloc[:,1])))

print("")

print("Age test RMSE:" ,np.sqrt(mean_squared_error(y_test.iloc[:,0], ypred_X_test.iloc[:,0])))
print("Capacity test RMSE:" , np.sqrt(mean_squared_error(y_test.iloc[:,1], ypred_X_test.iloc[:,1])))



#mean_absolute_error

#print("Age test MAE:", mean_absolute_error(y_test.iloc[:,0], ypred_X_test.iloc[:,0]))
#print("Capacity test MAE:", mean_absolute_error(y_test.iloc[:,1], ypred_X_test.iloc[:,1])+1)


#print("Age train MAE:" ,mean_absolute_error(y_train.iloc[:,0], ypred_X_train.iloc[:,0]))
#print("Capacity train MAE:" , mean_absolute_error(y_train.iloc[:,1], ypred_X_train.iloc[:,1])*10)

end_time = time.time()
time_taken = end_time - start_time
print("Execution Time : ",time_taken/60)
#ypred = model.predict(X_test)
#ypred_X_test = pd.DataFrame(ypred_X_test)


#print("y1 MSE:%.4f" % np.sqrt(mean_squared_error(y_test.iloc[:,0], ypred_X_test.iloc[:,0])))
#print("y2 MSE:%.4f" % np.sqrt(mean_squared_error(y_test.iloc[:,1], ypred_X_test.iloc[:,1])))
#print("Execution Time : ",time_taken/60)








test = pd.DataFrame(columns = ['voltage_min','soc','cycle_count','temperature_max'])
test= test.append({'voltage_min':3.0016,'soc':20,'cycle_count':2800,'temperature_max':48.6},ignore_index=True)

y_pred_test = Ensemble_model.predict(test)
preds_test = pd.DataFrame(y_pred_test)
preds_test = preds_test.iloc[:,:]

y_test_df = pd.DataFrame(y_test)
y_test_df = y_test_df.reset_index()
y_test_df = y_test_df.drop(columns = 'cycle_count')
y_test_loc = y_test_df.iloc[:,:]


#x_ax = range(0,500)
plt.figure(figsize = (10,5))
plt.scatter(X_train.cycle_count, y_train.iloc[:,0], label="Age-test", color='black',s=0.3)
plt.scatter(test.cycle_count, preds_test.iloc[:,0], label="Age-pred", color='red',s=20)
plt.ylabel("Age")
plt.xlabel("Cycles")
plt.title("Age: Actual vs Predicted")
plt.legend()
plt.show() 


#x_ax = range(0,500)
plt.figure(figsize = (10,5))
plt.scatter(X_train.cycle_count, y_train.iloc[:,1], label="Capacity-test", color='black',s=0.3)
plt.scatter(test.cycle_count, preds_test.iloc[:,1], label="Capacity-pred", color='red',s=20)
plt.ylabel("Capacity")
plt.xlabel("Cycles")
plt.title("Capacity: Actual vs Predicted")
plt.legend()
plt.show() 


