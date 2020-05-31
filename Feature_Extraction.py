import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


datadf = pd.read_csv("BatteryModelling.csv")
df=datadf.loc[:,['voltage', 'current', 'soc', 'temperature','cycle_count','capacity','age']]

def runningmean(x, N):
    # Initilize placeholder array
    y = np.zeros((len(x) - (N - 1),))
    for i in range(len(x) - (N - 1)):
         y[i] = np.mean(x[i:(i + N)])
    return y
def runningmax(x, N):
    # Initilize placeholder array
    y = np.zeros((len(x) - (N - 1),))
    for i in range(len(x) - (N - 1)):
         y[i] = np.max(x[i:(i + N)])
    return y
def runningmin(x, N):
    # Initilize placeholder array
    y = np.zeros((len(x) - (N - 1),))
    for i in range(len(x) - (N - 1)):
         y[i] = np.min(x[i:(i + N)])
    return y

def runningrms(x, N):
    # Initilize placeholder array
    y = np.zeros((len(x) - (N - 1),))
    for i in range(len(x) - (N - 1)):
         y[i] = np.sqrt(np.mean(np.power((x[i:(i + N)]),2)))
    return y


fe = pd.DataFrame()
N = 100
rmean = pd.DataFrame()
rmax = pd.DataFrame()
rmin = pd.DataFrame()
rrms = pd.DataFrame()
'''
for i in range(0,len(df.columns)):
    m = df.iloc[:,i]
    r_mean = runningmean(m, N)
    rmean = rmean.append([r_mean])
    r_max = runningmax(m, N)
    rmax = rmax.append([r_max])
    r_min = runningmin(m, N)
    rmin = rmin.append([r_min])
    r_rms = runningrms(m, N)
    rrms = rrms.append([r_rms])
   
    result_td = pd.DataFrame({'mean':[rmean],'max':[rmax],'min':[rmin],'rms':[rrms]})
    fe = pd.concat([fe,result_td])
    
rmean.reset_index(inplace = True,drop = True) 
rmean = rmean.T 
rmean.columns = ['voltage', 'current', 'soc', 'temperature','cycle_count','capacity','age']

rmax.reset_index(inplace = True,drop = True) 
rmax = rmax.T 
rmax.columns = ['voltage_max', 'current_max', 'soc_max', 'temperature_max','cycle_count_max','capacity_max','age_max']


rmin.reset_index(inplace = True,drop = True) 
rmin = rmin.T 
rmin.columns = ['voltage_min', 'current_min', 'soc_min', 'temperature_min','cycle_count_min','capacity_min','age_min']

rrms.reset_index(inplace = True,drop = True) 
rrms = rrms.T 
rrms.columns = ['voltage_rms', 'current_rms', 'soc_rms', 'temperature_rms','cycle_count_rms','capacity_rms','age_rms']


data = pd.concat([rmean,rmax,rmin,rrms ], axis=1)

data.to_csv("C:\\Users\\mouna\\Documents\\Battery_project\\Battery_Features.csv")
print("Done")   

'''


data = pd.read_csv("Battery_Features.csv")

data.columns

plt.figure(figsize=(10,5))
plt.plot(data["voltage"],label='data')
plt.title("voltage")
plt.xlabel("Time in minutes")
plt.ylabel("voltage")
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(data["soc_min"],label='data')
plt.title("SOC")
plt.xlabel("Time in minutes")
plt.ylabel("SOC")
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(data["temperature"],label='data')
plt.title("temperature")
plt.xlabel("Time in minutes")
plt.ylabel("temperature")
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(data["voltage_min"],label='data')
plt.title("voltage_min")
plt.xlabel("Time in minutes")
plt.ylabel("voltage_min")
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(data["temperature_max"],label='data')
plt.title("temperature_max")
plt.xlabel("Time in minutes")
plt.ylabel("SOC")
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(data["capacity"],label='data')
plt.title("capacity")
plt.xlabel("Time in minutes")
plt.ylabel("capacity")
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(data["age"],label='data')
plt.title("age")
plt.xlabel("Time in minutes")
plt.ylabel("age")
plt.legend()
plt.show()
