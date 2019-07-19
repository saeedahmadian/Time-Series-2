import os
import pandas as pd
import numpy as np
def get_excel_content(NameofFolder):
    os.chdir(os.getcwd()+'\\'+NameofFolder)
    excel_list = [i for i in os.listdir(os.getcwd()) if i.endswith('.xls')]
    return excel_list

def process_excel_sheets(Nameofexcel):
    excel_file=pd.ExcelFile(Nameofexcel)
    sheet_dict={}
    for i,sheet in enumerate(excel_file.sheet_names):
        if i!=0:
            df=excel_file.parse(sheet,header=0,index_col='Date')
            df.index=pd.to_datetime(df.index)+ pd.to_timedelta(df.Hour, unit='h')
            sheet_dict[i]=df
    return sheet_dict

def gather_info(ExcelList,Nofsheerts):
    info=dict.fromkeys(np.arange(1,Nofsheerts+1),[])
    for excel in ExcelList:
        print(excel)
        tmp=process_excel_sheets(excel)
        for key, val in tmp.items():
            info[key].append(val)
    return info
