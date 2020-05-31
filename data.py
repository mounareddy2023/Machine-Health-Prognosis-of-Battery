import pandas as pd

#import os
#os.chdir('C:\\Users\\mouna\\OneDrive\\Documents\\Battery_project\\Datasets\\temp')


dataset = pd.read_csv('RawData.csv')
ds=dataset.describe()
ds.to_csv("Describe_rawdata.csv")
df = pd.DataFrame(dataset)


cycle_count=[]
cycle = []

for i in range(0,len(df["voltage"])):
    
    if df["current"][i] == 20 :
       cycle.append("discharge")
    else :
       cycle.append("charge")
    
df['cycle'] = cycle

dat = df.loc[df.cycle == 'charge',]

test_list = df.loc[df.cycle == 'charge',].index
res = [test_list[i+1] - test_list[i] for i in range(len(test_list)-1)]
res.insert(0,0)
dat['diff']=res
ind = dat.loc[dat['diff']>1,].index

start = 0
count = 1
for i in range(0,len(ind)):
    df.loc[start:(ind[i]-1),'cycle_count'] = count
    count = count+1
    start=ind[i]

df.loc[ind[i]:,'cycle_count']=count

'''
State = []
for i in range(0,len(df["cycle_count"])):
    
    if df["cycle_count"][i] > 0 and df["cycle_count"][i] <= 750 :
       State.append("Best State")
    elif df["cycle_count"][i] <= 1500 :
       State.append("Better State")
    elif df["cycle_count"][i] <= 2250 :
       State.append("Good State")
    elif df["cycle_count"][i] <= 2940 :
       State.append("Average State")
    elif df["cycle_count"][i] <= 2990 :
       State.append("Battery Critical")
    elif df["cycle_count"][i] > 2990 and df["cycle_count"][i] <= 3000 :
       State.append("Replace Battery")
    else:
       State.append("Battery Dead")
       
df['State'] = State
'''

discharge = df[df['cycle']=='discharge']
dis=discharge.describe()
dis.to_csv("Describe_discharge_rawdata.csv")

#df.to_csv("file:///C:/Users/mouna/Documents/Battery_project/Datasets/temp/BatteryDischargeClass.csv")
#discharge.to_csv("file:///C:/Users/mouna/Documents/Battery_project/Datasets/temp/BatteryModelling.csv")

print("CSV file created")
