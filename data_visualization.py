import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir('C:\\Users\\mouna\\OneDrive\\Documents\\Battery_project\\Datasets\\temp')


dataset_25d = pd.read_csv('ScopeData_20soc_25d.csv')
df_25d = pd.DataFrame(dataset_25d)

dataset_40d = pd.read_csv('ScopeData_20soc_60d.csv')
df_40d = pd.DataFrame(dataset_40d)


print(df_25d.describe())
print(df_40d.describe())


print(df_25d.corr())
print(df_40d.corr())


df_25d1=df_25d.iloc[:,:]
df_40d1=df_40d.iloc[:,:]





fig, ax = plt.subplots(figsize = (10,5))
plt.plot(df_25d1['voltage'],color='blue', label='25d')
plt.plot(df_40d1['voltage'],linestyle='-.',color='red', label='60d')
plt.ylabel('Voltage')
plt.xlabel("Time in seconds")
plt.title('Voltage variation depending on temperature at 80% DOD')
plt.legend()
plt.grid()
plt.show()

fig, ax = plt.subplots(figsize = (10,5))
plt.plot(df_25d1['temperature'],color='blue', label='25d')
plt.plot(df_40d1['temperature'],linestyle='-.',color='red', label='60d')
plt.ylabel('Temperature')
plt.xlabel("Time in seconds")
plt.title('Charging Varying temperature')
plt.legend()
plt.grid()
plt.show()

fig, ax = plt.subplots(figsize = (10,5))
plt.plot(df_25d1['current'],color='blue', label='25d')
plt.plot(df_40d1['current'],linestyle='-.',color='red', label='60d')
plt.ylabel('Current')
plt.xlabel("Time in seconds")
plt.title('Current variation depending on temperature at 80% DOD')
plt.legend()
plt.grid()
plt.show()

fig, ax = plt.subplots(figsize = (10,5))
plt.plot(df_25d1['soc'],color='blue', label='25d')
plt.plot(df_40d1['soc'],linestyle='-.',color='red', label='60d')
plt.ylabel('SOC')
plt.xlabel("Time in seconds")
plt.title('SOC variation depending on temperature at 80% DOD')
plt.legend()
plt.grid()
plt.show()

fig, ax = plt.subplots(figsize = (10,5))
plt.plot(df_25d1['age'],color='blue', label='25d')
plt.plot(df_40d1['age'],linestyle='-.',color='red', label='60d')
plt.ylabel('Age')
plt.xlabel("Time in seconds")
plt.title('Age variation depending on temperature at 80% DOD')
plt.legend()
plt.grid()
plt.show()

fig, ax = plt.subplots(figsize = (10,5))
plt.plot(df_25d1['capacity'],color='blue', label='25d')
plt.plot(df_40d1['capacity'],linestyle='-.',color='red', label='60d')
plt.ylabel('Capacity')
plt.xlabel("Time in seconds")
plt.title('Capacity variation depending on temperature at 80% DOD')
plt.legend()
plt.grid()
plt.show()






dataset_25d = pd.read_csv('ScopeData_80soc.csv')
df_25d = pd.DataFrame(dataset_25d)

dataset_40d = pd.read_csv('ScopeData_80soc_60d.csv')
df_40d = pd.DataFrame(dataset_40d)

df_25d2=df_25d.iloc[:,:]
df_40d2=df_40d.iloc[:,:]

fig, ax = plt.subplots(figsize = (10,5))
plt.plot(df_25d2['voltage'],color='blue', label='25d')
plt.plot(df_40d2['voltage'],linestyle='-.',color='red', label='60d')
plt.ylabel('Voltage')
plt.xlabel("Time in seconds")
plt.title('Voltage variation depending on temperature at 20% DOD')
plt.legend()
plt.grid()
plt.show()

fig, ax = plt.subplots(figsize = (10,5))
plt.plot(df_25d2['temperature'],color='blue', label='25d')
plt.plot(df_40d2['temperature'],linestyle='-.',color='red', label='60d')
plt.ylabel('Temperature')
plt.xlabel("Time in seconds")
plt.title('Varying temperature')
plt.legend()
plt.grid()
plt.show()

fig, ax = plt.subplots(figsize = (10,5))
plt.plot(df_25d2['current'],color='blue', label='25d')
plt.plot(df_40d2['current'],linestyle='-.',color='red', label='60d')
plt.ylabel('Current')
plt.xlabel("Time in seconds")
plt.title('Current variation depending on temperature at 20% DOD')
plt.legend()
plt.grid()
plt.show()

fig, ax = plt.subplots(figsize = (10,5))
plt.plot(df_25d2['soc'],color='blue', label='25d')
plt.plot(df_40d2['soc'],linestyle='-.',color='red', label='60d')
plt.ylabel('SOC')
plt.xlabel("Time in seconds")
plt.title('SOC variation depending on temperature at 20% DOD')
plt.legend()
plt.grid()
plt.show()

fig, ax = plt.subplots(figsize = (10,5))
plt.plot(df_25d1['age'],color='blue', label='25d')
plt.plot(df_40d1['age'],linestyle='-.',color='red', label='60d')
plt.ylabel('Age')
plt.xlabel("Time in seconds")
plt.title('Age variation depending on temperature at 20% DOD')
plt.legend()
plt.grid()
plt.show()

fig, ax = plt.subplots(figsize = (10,5))
plt.plot(df_25d1['capacity'],color='blue', label='25d')
plt.plot(df_40d1['capacity'],linestyle='-.',color='red', label='60d')
plt.ylabel('Capacity')
plt.xlabel("Time in seconds")
plt.title('Capacity variation depending on temperature at 20% DOD')
plt.legend()
plt.grid()
plt.show()




#------------------------------------------------------------------------------

dataset_25d_20soc = pd.read_csv('ScopeData_20soc_25d.csv')
df_25d_20soc = pd.DataFrame(dataset_25d_20soc)

dataset_25d_80soc = pd.read_csv('ScopeData_80soc.csv')
df_25d_80soc = pd.DataFrame(dataset_25d_80soc)


df_25d1_20soc=df_25d_20soc.iloc[:,:]
df_25d1_80soc=df_25d_80soc.iloc[:,:]



fig, ax = plt.subplots(figsize = (10,5))
plt.plot(df_25d1_20soc['voltage'],color='blue', label='80% DOD')
plt.plot(df_25d1_80soc['voltage'],linestyle='-.',color='red', label='20% DOD')
plt.ylabel('Voltage')
plt.xlabel("Time in seconds")
plt.title('Voltage variation depending on SoC at 80% DOD and at 20% DOD')
plt.legend()
plt.grid()
plt.show()

fig, ax = plt.subplots(figsize = (10,5))
plt.plot(df_25d1_20soc['temperature'],color='blue', label='80% DOD')
plt.plot(df_25d1_80soc['temperature'],linestyle='-.',color='red', label='20% DOD')
plt.ylabel('Temperature')
plt.xlabel("Time in seconds")
plt.title('Temperature variation depending on SoC at 80% DOD and at 20% DOD')
plt.legend()
plt.grid()
plt.show()

fig, ax = plt.subplots(figsize = (10,5))
plt.plot(df_25d1_20soc['current'],color='blue', label='80% DOD')
plt.plot(df_25d1_80soc['current'],linestyle='-.',color='red', label='20% DOD')
plt.ylabel('Current')
plt.xlabel("Time in seconds")
plt.title('Current variation depending on SoC at 80% DOD and at 20% DOD')
plt.legend()
plt.grid()
plt.show()

fig, ax = plt.subplots(figsize = (10,5))
plt.plot(df_25d1_20soc['soc'],color='blue', label='80% DOD')
plt.plot(df_25d1_80soc['soc'],linestyle='-.',color='red', label='20% DOD')
plt.ylabel('SOC')
plt.xlabel("Time in seconds")
plt.title('SOC variation depending on SoC at 80% DOD and at 20% DOD')
plt.legend()
plt.grid()
plt.show()

fig, ax = plt.subplots(figsize = (10,5))
plt.plot(df_25d1_20soc['age'],color='blue', label='80% DOD')
plt.plot(df_25d1_80soc['age'],linestyle='-.',color='red', label='20% DOD')
plt.ylabel('Age')
plt.xlabel("Time in seconds")
plt.title('Age variation depending on SoC at 80% DOD and at 20% DOD')
plt.legend()
plt.grid()
plt.show()

fig, ax = plt.subplots(figsize = (10,5))
plt.plot(df_25d1_20soc['capacity'],color='blue', label='80% DOD')
plt.plot(df_25d1_80soc['capacity'],linestyle=':',color='blue', label='20% DOD')
plt.ylabel('Capacity')
plt.xlabel("Time in seconds")
plt.title('Capacity variation depending on SoC at 80% DOD and at 20% DOD')
plt.legend()
plt.grid()
plt.show()

