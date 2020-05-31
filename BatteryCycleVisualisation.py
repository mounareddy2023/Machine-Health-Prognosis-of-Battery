
import pandas as pd
import matplotlib.pyplot as plt
#import os
#os.chdir('C:\\Users\\mouna\\OneDrive\\Documents\\Battery_project\\Datasets\\temp')


dataset = pd.read_csv('BatteryModelling.csv')
df = pd.DataFrame(dataset)

cycle_1 = df[df['cycle_count']==10]
cycle_1.reset_index(inplace = True)

cycle_500 = df[df['cycle_count']==500]
cycle_500.reset_index(inplace = True)

cycle_1000 = df[df['cycle_count']==1000]
cycle_1000.reset_index(inplace = True)

cycle_1500 = df[df['cycle_count']==1500]
cycle_1500.reset_index(inplace = True)

cycle_2000 = df[df['cycle_count']==2000]
cycle_2000.reset_index(inplace = True)

cycle_2500 = df[df['cycle_count']==2500]
cycle_2500.reset_index(inplace = True)

cycle_3000 = df[df['cycle_count']==3000]
cycle_3000.reset_index(inplace = True)



plt.figure(figsize=(10,5))
plt.plot(cycle_1["soc"],label='cycle_1')
plt.plot(cycle_500["soc"],label='cycle_500')
plt.plot(cycle_1000["soc"],label='cycle_1000')
plt.plot(cycle_1500["soc"],label='cycle_1500')
plt.plot(cycle_2000["soc"],label='cycle_2000')
plt.plot(cycle_3000["soc"],label='cycle_3000')
plt.title("SOC at differnt discharge cycles ")
plt.xlabel("Time in minutes")
plt.ylabel("SOC")
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(cycle_1["voltage"],label='cycle_1')
plt.plot(cycle_500["voltage"],label='cycle_500')
plt.plot(cycle_1000["voltage"],label='cycle_1000')
plt.plot(cycle_1500["voltage"],label='cycle_1500')
plt.plot(cycle_2000["voltage"],label='cycle_2000')
plt.plot(cycle_3000["voltage"],label='cycle_3000')
plt.title("Voltage at differnt discharge cycles ")
plt.xlabel("Time in minutes")
plt.ylabel("Voltage")
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(cycle_1["temperature"],label='cycle_1')
plt.plot(cycle_500["temperature"],label='cycle_500')
plt.plot(cycle_1000["temperature"],label='cycle_1000')
plt.plot(cycle_1500["temperature"],label='cycle_1500')
plt.plot(cycle_2000["temperature"],label='cycle_2000')
plt.plot(cycle_3000["temperature"],label='cycle_3000')
plt.title("Temperature at differnt discharge cycles ")
plt.xlabel("Time in minutes")
plt.ylabel("Temperature")
plt.legend()
plt.show()



'''
import seaborn as sb
sb.pairplot(df)
from autoviz.AutoViz_Class import AutoViz_Class
AV = AutoViz_Class()
data = df.drop(['Time'],axis=1)
sep = ','
dft = AV.AutoViz(filename="",sep=sep, depVar=None, dfte=data, header=0, verbose=0, 
                     lowess=False, chart_format='svg', max_rows_analyzed=254030, max_cols_analyzed=7)
'''
'''

plt.figure(figsize=(10,5))
plt.plot(cycle_1["age"],label='cycle_1')
plt.plot(cycle_500["age"],label='cycle_500')
plt.plot(cycle_1000["age"],label='cycle_1000')
plt.plot(cycle_1500["age"],label='cycle_1500')
plt.plot(cycle_2000["age"],label='cycle_2000')
plt.plot(cycle_3000["age"],label='cycle_3000')
plt.title("Age at differnt discharge cycles ")
plt.xlabel("Time in minutes")
plt.ylabel("age")
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(cycle_1["capacity"],label='cycle_1')
plt.plot(cycle_500["capacity"],label='cycle_500')
plt.plot(cycle_1000["capacity"],label='cycle_1000')
plt.plot(cycle_1500["capacity"],label='cycle_1500')
plt.plot(cycle_2000["capacity"],label='cycle_2000')
plt.plot(cycle_3000["capacity"],label='cycle_3000')
plt.title("Capacity at differnt discharge cycles ")
plt.xlabel("Time in minutes")
plt.ylabel("capacity")
plt.legend()
plt.show()

'''


'''
dataset = pd.read_csv('LiBatteryCycle.csv')
df = pd.DataFrame(dataset)

charge = df[df['cycle']=='charge']
charge.describe()

ccycle_1 = charge[charge['cycle_count']==10]
ccycle_1.reset_index(inplace = True)

ccycle_500 = charge[charge['cycle_count']==500]
ccycle_500.reset_index(inplace = True)

ccycle_1000 = charge[charge['cycle_count']==1000]
ccycle_1000.reset_index(inplace = True)

ccycle_1500 = charge[charge['cycle_count']==1500]
ccycle_1500.reset_index(inplace = True)

ccycle_2000 = charge[charge['cycle_count']==2000]
ccycle_2000.reset_index(inplace = True)

ccycle_2500 = charge[charge['cycle_count']==2500]
ccycle_2500.reset_index(inplace = True)

ccycle_3000 = charge[charge['cycle_count']==3000]
ccycle_3000.reset_index(inplace = True)

ccycle_3500 = charge[charge['cycle_count']==3500]
ccycle_3500.reset_index(inplace = True)



plt.plot(ccycle_1["soc"],label='cycle_1')
plt.plot(ccycle_500["soc"],label='cycle_500')
plt.plot(ccycle_1000["soc"],label='cycle_1000')
plt.plot(ccycle_1500["soc"],label='cycle_1500')
plt.plot(ccycle_2000["soc"],label='cycle_2000')
plt.plot(ccycle_3000["soc"],label='cycle_3000')
plt.title("soc")
plt.legend()
plt.show()



plt.plot(ccycle_1["voltage"],label='cycle_1')
plt.plot(ccycle_500["voltage"],label='cycle_500')
plt.plot(ccycle_1000["voltage"],label='cycle_1000')
plt.plot(ccycle_1500["voltage"],label='cycle_1500')
plt.plot(ccycle_2000["voltage"],label='cycle_2000')
plt.plot(ccycle_3000["voltage"],label='cycle_3000')
plt.title("Voltage")
plt.legend()
plt.show()


plt.plot(ccycle_1["temperature"],label='cycle_1')
plt.plot(ccycle_500["temperature"],label='cycle_500')
plt.plot(ccycle_1000["temperature"],label='cycle_1000')
plt.plot(ccycle_1500["temperature"],label='cycle_1500')
plt.plot(ccycle_2000["temperature"],label='cycle_2000')
plt.plot(ccycle_3000["temperature"],label='cycle_3000')
plt.title("temperature")
plt.legend()
plt.show()


plt.plot(ccycle_1["age"],label='cycle_1')
plt.plot(ccycle_500["age"],label='cycle_500')
plt.plot(ccycle_1000["age"],label='cycle_1000')
plt.plot(ccycle_1500["age"],label='cycle_1500')
plt.plot(ccycle_2000["age"],label='cycle_2000')
plt.plot(ccycle_3000["age"],label='cycle_3000')
plt.title("age")
plt.legend()
plt.show()


plt.plot(ccycle_1["capacity"],label='cycle_1')
plt.plot(ccycle_500["capacity"],label='cycle_500')
plt.plot(ccycle_1000["capacity"],label='cycle_1000')
plt.plot(ccycle_1500["capacity"],label='cycle_1500')
plt.plot(ccycle_2000["capacity"],label='cycle_2000')
plt.plot(ccycle_3000["capacity"],label='cycle_3000')
plt.title("capacity")
plt.legend()
plt.show()
'''