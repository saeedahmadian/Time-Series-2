import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm

from read_excel import *

def TimeSerieOLS(TS,lag):
    TS_lag = TS.shift(lag)
    TS_lag = TS_lag.dropna()
    TS = TS.iloc[lag:,:]
    res = []
    newdata=[]
    for col in TS.columns:
        res.append(sm.OLS(TS[col], TS_lag[col]).fit())
        tmp = pd.DataFrame(data=sm.OLS(TS[col], TS_lag[col]).fit().predict(), index=TS[col].index,columns=[col+'_prediction'])
        tmp[col]=TS[col]
        tmp[col +'_prediction_error'] = tmp[col]-tmp[col + '_prediction']
        newdata.append(tmp)
    return res,newdata


def ErrorOLS(err,lag):
    err=pd.DataFrame(data=err,columns=[err.name])
    err_lag = err.shift(lag).dropna()
    err = err.iloc[lag:,:]
    model = []
    res = []
    for col in err.columns:
        model.append(sm.OLS(err[col],err_lag[col]))
        res.append(sm.OLS(err[col],err_lag[col]).fit())
    return res

All_excels= get_excel_content('DATA')
# Initialize the Data dictionary
#tmp=gather_info(All_excels,9)

All_Data={}
for i in range(1,10):
    All_Data[i]=[]

for i in All_excels:
    print(i)
    tmp_dict= process_excel_sheets(i)
    for key,val in tmp_dict.items():
        All_Data[key].append(val)

DataBase=dict.fromkeys(np.arange(1,10),None)
for key,_ in DataBase.items():
    DataBase[key]=pd.concat(All_Data[key][:])

All_TSs=DataBase
All_zones=[]
for i, (key, val) in enumerate(DataBase.items()):
    # figure_list = plt.figure(figsize=[18, 10])
    # figure_list.suptitle(zone_names[i])
    df = pd.DataFrame(index=DataBase[key].index)
    df['Day-Ahead_Demand'] = DataBase[key].iloc[:, 1]
    df['Real-time_Demand'] = DataBase[key].iloc[:, 2]
    df['Day-Ahead_Price'] = DataBase[key].iloc[:, 3]
    df['Real-time_Price'] = DataBase[key].iloc[:, 7]
    df_daily = df.resample('D').mean()
    df_weekly = df.resample('W').mean()
    df_monthly = df.resample('M').mean()
    df_list = [df, df_daily, df_weekly, df_monthly]
    All_zones.append(df_list)


counter=0
zone_names=['ISONE_CA', 'Portland','Concord','Burlington','Bridgeport','Providence','SEMASS','Worcester','Boston']
for zone_info in All_zones:
    Daily_Info=zone_info[1]
    TSOLS, process_data = TimeSerieOLS(Daily_Info,1)
    for data in process_data:
        err_analysis = ErrorOLS(data.iloc[:,-1],1)[0]
        name=zone_names[counter]+'_'+data.columns[-1]
        f = open(name+'.txt','w')
        f.write(str(err_analysis.summary()))
        f.close()

    counter+=1

