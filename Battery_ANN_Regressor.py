import numpy as np
import pandas as pd
from keras.layers import Dense, Activation
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# Importing the dataset

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
df_train = data.loc[:,['voltage_min','current','soc', 'temperature_max','capacity','age']]
df_test = df_train.iloc[1000:2100,:]


X_train,y_train = df_train.loc[:,['voltage_min','current','soc', 'temperature_max']],df_train.loc[:,['age','capacity']]

X_test,y_test = df_test.loc[:,['voltage_min','current','soc', 'temperature_max']],df_test.loc[:,['age','capacity']]

'''
X = dataset.loc[:, ['voltage', 'soc', 'temperature']]
y = dataset.loc[:, ['age']]

# Splitting the dataset into the Training set and Test set
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)


X_train = X.iloc[:,:]
y_train = y.iloc[:,:]
X_test = X.iloc[1:100000,:]
y_test = y.iloc[1:100000,:]
'''
'''
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()




X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''

import time
start_time = time.time()
# Initialising the ANN
model = Sequential()

# Adding the input layer and the first hidden layer
model.add(Dense(32, activation = 'relu', input_dim = 4))

# Adding the second hidden layer
model.add(Dense(units = 32, activation = 'relu'))

# Adding the third hidden layer
model.add(Dense(units = 32, activation = 'relu'))

# Adding the output layer

model.add(Dense(units = 2))

#model.add(Dense(1))
# Compiling the ANN
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the ANN to the Training set
model.fit(X_train, y_train, batch_size = 10, epochs = 100)


end_time = time.time()
time_taken = end_time - start_time
#print("Execution Time : ",time_taken/60)


from sklearn.metrics import mean_squared_error,explained_variance_score

y_pred_train = model.predict(X_train)
preds_train = pd.DataFrame(y_pred_train)
preds_train = preds_train.iloc[:,:]



'''

mse=mean_squared_error(y_train, y_pred_train)
print("MSE (train): %f" % (mse))
rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
print("RMSE (train): %f" % (rmse))
#from sklearn.metrics import explained_variance_score
print('Accuracy (train):',np.abs(explained_variance_score(preds_train,y_train)*100))


'''

y_pred_test = model.predict(X_test)
preds_test = pd.DataFrame(y_pred_test)
preds_test = preds_test.iloc[:,:]

y_test_df = pd.DataFrame(y_test)
y_test_df = y_test_df.reset_index()
y_test_df = y_test_df.drop(columns = 'cycle_count')
y_test_loc = y_test_df.iloc[:,:]



ypred_X_test = model.predict(X_test)
ypred_X_test = pd.DataFrame(ypred_X_test)

ypred_X_train = model.predict(X_train)
ypred_X_train = pd.DataFrame(ypred_X_train)

y_test = pd.DataFrame(y_test)




#y_pred_train = model.predict(X_train)
preds_train = pd.DataFrame(ypred_X_train)
preds_train = preds_train.iloc[:,:]




#x_ax = range(0,500)
plt.figure(figsize = (10,5))
plt.plot( y_train.iloc[:,0], label="Age-train", color='black')
plt.plot( ypred_X_train.iloc[:,0], label="Age-pred", color='red')
plt.xlabel("Cycles")
plt.ylabel("Age")
plt.title("Age: Actual vs Predicted")
plt.legend()
plt.show() 


plt.figure(figsize = (10,5))
plt.plot( y_train.iloc[:,1], label="Capacity-train", color='black')
plt.plot( ypred_X_train.iloc[:,1], label="Capacity-pred", color='red')
plt.xlabel("Cycles")
plt.ylabel("Capacity")
plt.title("Capacity: Actual vs Predicted")
plt.legend()
plt.show() 



#ypred = model.predict(X_test)
#ypred_X_test = pd.DataFrame(ypred_X_test)


print("y1 MSE:%.4f" % np.sqrt(mean_squared_error(y_test.iloc[:,0], ypred_X_test.iloc[:,0])))
print("y2 MSE:%.4f" % np.sqrt(mean_squared_error(y_test.iloc[:,1], ypred_X_test.iloc[:,1])))
print("Execution Time : ",time_taken/60)
'''


mse=mean_squared_error(y_test, y_pred_test)
print("MSE (test): %f" % (mse))
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
print("RMSE (test): %f" % (rmse))
print('Accuracy (test):',np.abs(explained_variance_score(y_pred_test,y_test)*100))
'''