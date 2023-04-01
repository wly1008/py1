# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 19:57:36 2023

@author: wly
"""

import pandas as pd
import os
import re

#创建文件夹
def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
	else:
		pass

def read_data(file_paths):
    dfs = []
    for file_path in file_paths:
        df = pd.read_csv(file_path, header=None)
        dfs.append(df)
    return pd.concat(dfs)

def get_num(x):
    
    data = re.findall(r'-?\d+',x)
    
    return pd.Series(data,index=range(1,len(data)+1))
    
def process_data(x):
    df = pd.DataFrame()
    df[['site', 'year', 'month', 'variable', 'data']] = x[0].str.extract(r'(\S{11})(\d{4})(\d{2})(\S{4})(.*)') # <<<<<正则表达式
    
    df[[i for i in range(1,32)]] = df['data'].apply(get_num)  # <<<<<<<<<<取出数字
    
    # 获得月天数
    df['date'] = pd.to_datetime(df['year'] + '-' + df['month']) 
    df['day'] = df['date'].dt.daysinmonth
    return df.drop(['date','data'], axis=1)

# 分组各站点数据合并
def func_site(x):
    df = pd.DataFrame()
    gros = x.groupby('site')
    for Site,gro_data in gros:

        # 按月天数合并年数据
        df_all = pd.Series()
        for i in range(len(gro_data)):
            
            ser = gro_data.iloc[i]
            day = ser.day
            dfn = ser.loc[1:day]
            df_all = pd.concat([df_all,dfn])
        # 重设索引
        cols = iter(range(1,len(df_all)+1))
        df_all = df_all.rename(index=lambda x: next(cols))
        # 加上基础属性
        df_all = pd.concat([ser[['site', 'variable', 'year']],df_all])
        # 合并各站点数据
        df = pd.concat([df,df_all])
    return df

if __name__ == '__main__':
    # 输入、输出
    file_paths = [r'E:\Python\pythonlx\8 dly处理\AM000037874.dly', r'E:\Python\pythonlx\8 dly处理\AM000037880.dly']
    output_dir = r'E:\Python\pythonlx\8 dly处理\结果'
    
    
    df = read_data(file_paths)
    df_day = process_data(df)
    # 按年份、变量分组
    gros = df_day.groupby(['year','variable'])
    
    for (Year,Var),gro_data in gros:

        df_out = func_site(gro_data).T
        
        # 输出
        file = os.path.join(output_dir, Var)
        mkdir(file)
        doc = f'{Var}_{Year}.txt'
        out_path = os.path.join(file, doc)
        
        # df_out.to_csv(out_path,sep='\t',index=0)
