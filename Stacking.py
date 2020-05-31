# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 23:15:51 2020

@author: mouna
"""
import pandas as pd

data = pd.read_csv("BatteryModelling.csv")

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
df_train = data.loc[:,['voltage_min','current','soc', 'temperature_max','capacity','age']]
df_test = df_train.iloc[1000:2100,:]
df = df_train.append(df_test , ignore_index = True)


X_train,Y_train = df_train.loc[:,['voltage_min','current','soc', 'temperature_max']],df_train.loc[:,['age']]

X_test,Y_test = df_test.loc[:,['voltage_min','current','soc', 'temperature_max']],df_test.loc[:,['age']]


#Feature Scaling
#________________
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
Y_train = Y_train.values.reshape((len(Y_train), 1)) #reshaping to fit the scaler
Y_train = sc.fit_transform(Y_train)
Y_train = Y_train.ravel()

import numpy as np
#RMLSE For Model Evaluation
def score(y_pred, y_true):
    error = np.square(np.log10(y_pred +1) - np.log10(y_true +1)).mean() ** 0.5
    score = 1 - error
    return score

actual_cost = list(df_test['age'])
actual_cost = np.asarray(actual_cost)


#--------------------------------------------------------------------------------



#eXtreme Gradient Boosting

############################################################################
#Importing and Initializing the Regressor
from xgboost import XGBRegressor
xgbr = XGBRegressor()
#Fitting the data to the regressor
xgbr.fit(X_train, Y_train)
#Predicting the Test set results
y_pred_xgbr = sc.inverse_transform(xgbr.predict(X_test))
#Evaluating
print("\n\nXGBoost SCORE : ", score(y_pred_xgbr, actual_cost))



#--------------------------------------------------------------------------------



#Random Forest Regression
############################################################################
#Importing and Initializing the Regressor
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
#Fitting the data to the regressor
rf.fit(X_train, Y_train)
#Predicting the Test set results
y_pred_rf = sc.inverse_transform(rf.predict(X_test))
#Evaluating
print("\n\nRandom Forest SCORE : ", score(y_pred_rf, actual_cost))




#--------------------------------------------------------------------------------



'''
#Linear Regression
############################################################################
#Importing and Initializing the Regressor
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
#Fitting the data to the regressor
lr.fit(X_train,Y_train)
#Predicting the Test set results
Y_pred_linear = sc.inverse_transform(lr.predict(X_test))
#Evaluating
print("\n\nLinear Regression SCORE : ", score(Y_pred_linear, actual_cost))

'''



#--------------------------------------------------------------------------------





#Stacking Ensemble Regression
###########################################################################
#Importing and Initializing the Regressor
from mlxtend.regressor import StackingCVRegressor

#Initializing Level One Regressorsxgbr = XGBRegressor()
#rf = RandomForestRegressor(n_estimators=100, random_state=1)
#lr = LinearRegression()

#Stacking the various regressors initialized before
stack = StackingCVRegressor(regressors=(xgbr ,rf),meta_regressor= xgbr, use_features_in_secondary=True)

#Fitting the data
stack.fit(X_train,Y_train)

#Predicting the Test set results
y_pred_ense = sc.inverse_transform(stack.predict(X_test))

#Evaluating
print("\n\nStackingCVRegressor SCORE : ", score(y_pred_ense, actual_cost))








