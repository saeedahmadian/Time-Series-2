import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose as decomp
import statsmodels.api as sm

from read_excel import *


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

def plot_decomposition(Decomp_TS):
    """
    """
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 8),sharex=True)

    ax1.plot(Decomp_TS.iloc[:,0], label=Decomp_TS.columns[0], color='black')
    ax1.legend(loc='best')
    ax1.tick_params(axis='x', rotation=45)

    ax2.plot(Decomp_TS.iloc[:,1], label='Trend Element', color='magenta')
    ax2.legend(loc='best')
    ax2.tick_params(axis='x', rotation=45)

    ax3.plot(Decomp_TS.iloc[:,2], label='Seasonality Element', color='green')
    ax3.legend(loc='best')
    ax3.tick_params(axis='x', rotation=45)

    ax4.plot(Decomp_TS.iloc[:,3], label='Residuals Element', color='red')
    ax4.legend(loc='best')
    ax4.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    TSname=Decomp_TS.columns[0]
    # Show graph
    # plt.suptitle('Trend, Seasonal, and Residual Decomposition of %s' % (TSname),
    #              x=0.5,
    #              y=1.05,
    #              fontsize=18)
    # plt.show()
    plt.savefig(TSname + '.png')
    plt.close()


def Decomposition(TS, nameOfTS, model_name='additive',frequency=None):
    decomposed= decomp(x=TS,model=model_name, freq= frequency)
    out= pd.DataFrame()
    out['Original_'+nameOfTS] = TS
    out["trend"] = decomposed.trend
    out["Seasonality"] = decomposed.seasonal
    out["residual"]= decomposed.resid
    return out

# feed the Daily Time Series to Transformation method and choosing only Real-time data
counter=0
zone_names=['ISONE_CA', 'Portland','Concord','Burlington','Bridgeport','Providence','SEMASS','Worcester','Boston']
for zone_info in All_zones:
    Daily_Info=zone_info[1]
    for col in Daily_Info.columns[0:3]:
        name = col + '_for_' + zone_names[counter]
        res= Decomposition(Daily_Info[col],name ,model_name='multiplicative',frequency=365)
        plot_decomposition(res)
    counter+=1







a=1