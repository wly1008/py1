# -*- coding: utf-8 -*-
"""
Created on Mon May  8 20:26:45 2023

@author: wly
"""

import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree as KDTree
    # http://docs.scipy.org/doc/scipy/reference/spatial.html

# nnan = []
def get_num(x,lst_and = []):   
    x = str(x) + '/'
    est = 'f'
    nums = []
    global nnan
    
    ls = [str(i) for i in range(10)] + lst_and
    for i in range(len(x)):
        n = x[i]
        if est == 'y':
            if not n in ls:
                end = i
                num = x[start:end]
                nums.append(num)
                est = 'f' 
                if not n in nnan:
                    nnan.append(n)
        elif n in ls:
            start = i
            est = 'y'           
        else:
            if not n in nnan:
                nnan.append(n)
            
    if len(nums) < 3:
        for i in range(3 - len(nums)):
            nums.append(0)
    return pd.Series(nums,index=range(1,len(nums)+1))


# def out_pkl():
    
#     with open(r'F:\PyCharm\pythonProject1\代码\mycode\1.pickle','wb') as f:
#         f.seek(0)  #定位
#         f.truncate()   #清空文件
#         data = [dict(globals())]
#         dill.dump(data,f)


# -----------------------------------------------------------------------------------------

def isiterable(x):
    try:
        iter(x)
        return True
    except:
        return False


def evals(*runs,**kwargs):
    """
    批量eval
    """
    returns = []
    for _run in runs:
        
        if isiterable(_run) & (type(_run) != str):
            # 递归
            returns.append([evals(i,**kwargs) for i in _run])
        else:
            # # 解包变量
            # for k,v in kwargs.items():
            #     locals()[k] = v
            
            
            # 操作
            returns.append(eval(_run,globals(),kwargs))
            # returns.append(eval(_run))

    if len(returns) == 1:
        returns = returns[0]
    return returns


def getattrs(src, *args, ds={},**kwargs):
    
    ds.update({'src':src})
    
    # ds,接收上一文件的(全局)变量,注销此段，则输入变量优先级更高。
    globals_old = set(ds)
    globals_now = set(globals())
    need_add = globals_old-globals_now
    data = {k:ds[k] for k in need_add}
    ds = data
    
    
    # 对需要的操作（args）进行整理
    runs = []
    for arg in args:

        while arg in kwargs:
            arg = kwargs[arg]
        
        if isiterable(arg) & (type(arg) != str):
            # 递归
            return getattrs(src, *arg,**kwargs)
        else:
            # 整理
            run = f'src.{arg}' if not ('src' in arg) else arg
            
            runs.append(run)
    
    # 调用批量操作函数
    return evals(*runs,**ds)









"""
字典相关函数
"""

# 查找
def findAll(target, dictData, notFound=[]):
    queue = [dictData]
    result = []
    while len(queue) > 0:
        data = queue.pop()
        for key, value in data.items():
            if key == target: result.append(value)
            elif type(value) == dict: queue.append(value)
    if not result: result = notFound
    return result

def find(target, dictData, notFound='没找到'):
    queue = [dictData]
    while len(queue) > 0:
        data = queue.pop()
        for key, value in data.items():
            if key == target: return value
            elif type(value) == dict: queue.append(value)
    return notFound



# 解包多重字典，未考虑重复项，如有重复取最深的一项
def ungroup(dictData,dtype = False):
    queue = [dictData]
    
    result = {}
    while len(queue) > 0:
        data = queue.pop()
        
        for key, value in data.items():
            
            if type(value) == dict: queue.append(value)
            elif dtype: 
                if type(value) == dtype: result[key] = value
            else: result[key] = value
    return result













































