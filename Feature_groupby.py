import matplotlib.pyplot as plt

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
print("Done")

import seaborn as sns

corr = data.corr()
#corr = abs(corr)
fig, ax = plt.subplots(figsize=(10, 10)) 

sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True,
    annot= True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=90,
    horizontalalignment='right'
);



data.columns

plt.figure(figsize=(10,5))
plt.plot(data["voltage"],label='data')
plt.title("Voltage")
plt.xlabel("Cycles")
plt.ylabel("Voltage")
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(data["voltage_min"],label='data')
plt.title("Voltage Minimum")
plt.xlabel("Cycles")
plt.ylabel("Voltage Minimum")
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(data["voltage_max"],label='data')
plt.title("Voltage Maximum")
plt.xlabel("Cycles")
plt.ylabel("Voltage Maximum")
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(data["temperature"],label='data')
plt.title("Temperature")
plt.xlabel("Cycles")
plt.ylabel("Temperature")
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(data["temperature_max"],label='data')
plt.title("Temperature Maximum")
plt.xlabel("Cycles")
plt.ylabel("Temperature Maximum")
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(data["capacity"],label='data')
plt.title("Capacity")
plt.xlabel("Cycles")
plt.ylabel("Capacity")
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(data["age"],label='data')
plt.title("Age")
plt.xlabel("Cycles")
plt.ylabel("Age")
plt.legend()
plt.show()
