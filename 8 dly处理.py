# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 21:04:03 2023

@author: wly
"""

import pandas as pd
import os
import re
import calendar
import numpy as np

def read_data(file_paths):
    return  pd.concat([pd.read_csv(file_path, header=None) for file_path in file_paths])

def get_num(x):
    
    data = re.findall(r'-?\d+',x)
    
    return pd.Series(data,index=range(1,len(data)+1))
    
def process_data(x):
    df = pd.DataFrame()
    df[['site', 'year', 'month', 'variable', 'data']] = x[0].str.extract(r'(\S{11})(\d{4})(\d{2})(\S{4})(.*)') # <<<<<正则表达式
    
    df[[i for i in range(1,32)]] = df['data'].apply(get_num)  # <<<<<<<<<<取出数字
    df.replace('-9999',np.nan,inplace = True)
    return df.drop(['data'], axis=1)

# 分组各站点数据合并
def func_site(x):
    day_iter = iter(range(1,366+1))# <<<<<<<<<<<年天数迭代器
    year = int(x.year.iat[0])# <<<<<<<<<<<年
    df = pd.DataFrame()
    
    # 按站点分组
    gros = x.groupby('site')
    for Site,gro_data in gros:

        # 按月天数合并年数据
        day_iter = iter(range(1,366+1))
        dfn_all = pd.DataFrame()
        for i in range(1,13):
            
            df_mon = gro_data[gro_data.month == '%.2d'%i]
            day = calendar.monthrange(year,i)[1]# <<<<<<<<<<<<<<<<<获得月天数
            dfn = df_mon.loc[:,1:day]
            for n in range(day):
                
                if n == 0:
                    mon_day = []
                mon_day.append(next(day_iter))# 该年第几天
            dfn.columns = mon_day
            try:
                dfn.index = [0]# <<<<<<<<<<<<<<<保证索引一致
            except:
                pass
            dfn_all = pd.concat([dfn_all,dfn],axis=1)

        # 加上基础属性
        # dfn_all = pd.concat([df_mon[['site', 'variable', 'year']],dfn_all])
        dfn_all['site'] = Site
        # 合并各站点数据
        df = pd.concat([df,dfn_all])
    return pd.concat([df.iloc[:,-1],df.iloc[:,:-1]],axis=1)

if __name__ == '__main__':
    # 输入、输出
    file_paths = [r'E:\Python\pythonlx\8 dly处理\AM000037874.dly', r'E:\Python\pythonlx\8 dly处理\AM000037880.dly']
    output_dir = r'E:\Python\pythonlx\8 dly处理\结果'
    
    
    df = read_data(file_paths)
    df_day = process_data(df)
    # 按年份、变量分组
    gros = df_day.groupby(['year','variable'])
    
    for (Year,Var),gro_data in gros:
        df_out = func_site(gro_data)
        
        # 输出
        file = os.path.join(output_dir, Var)
        if not (os.path.exists(file)):                   
            os.makedirs(file)
        doc = f'{Var}_{Year}.txt'
        out_path = os.path.join(file, doc)
        df_out.fillna('nan',inplace=True)
        df_out.to_csv(out_path,sep='\t',index=0,header=0)
        print(f'{Var}_{Year}')






