# -*- coding: utf-8 -*-
"""
Created on Mon May  8 20:26:45 2023

@author: wly
"""

import os
import numpy as np
import pandas as pd
import sys


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
    
    '''
    判断x能否迭代
    
    返回 bool类型
    '''
    try:
        iter(x)
        return True
    except:
        return False



def evals(*runs,**kwargs):
    '''
    批量操作函数

    Parameters
    ----------
    *runs : TYPE
        操作字符串集
    **kwargs : TYPE
        额外可用变量，默认可用此文件全局变量

    Returns
    -------
    returns : 
        结果列表,如果列表只有一个元素，则直接返回这个元素

    '''
    
    returns = []
    for run in runs:
        
        if isiterable(run) & (type(run) != str):
            # 递归
            returns.append([evals(i,**kwargs) for i in run])
        else:
            # # 解包变量
            # for k,v in kwargs.items():
            #     locals()[k] = v
            
            
            # 操作
            returns.append(eval(run,globals(),kwargs))
            # returns.append(eval(_run))

    if len(returns) == 1:
        returns = returns[0]
    return returns


def getattrs(src, *args, ds={}, get_runs=False, **kwargs):
    '''
    批量获得类属性

    Parameters
    ----------
    src : class
        被查询的类.
    *args : TYPE
        所需属性或函数（类中存在的，输入属性名、函数名即可）
    ds : TYPE, optional
        额外可用变量（优先级更高，可自行修改优先级），供操作函数使用，默认可用此文件全局变量. The default is {}.
    get_runs : TYPE, optional
        是否返回操作字符串列表. The default is False.
    **kwargs : 字典值获得对应属性所需操作，可为表达式，默认参数以字典形式写在“//ks//”之后，在ds中输入相应变量可替代默认参数
            非自身类函数调用时及自身在dic、kwargs中定义的属性调用时，src不可省略。
            必须使用src代表源数据。

            合并属性返回类型为list. e.g.'raster_size': ('height', 'width') -> [900, 600]
            如需特定属性请用函数. e.g. 'raster_size': r"(src.height, src.width)" or r"pd.Serise([src.height, src.width])"

    Returns
    -------
    if get_runs:返回操作字符串列表
    else:args对应属性值列表

    '''
    
    ds.update({'src':src})
    
    # ds,接收上一文件的(全局、参数)变量,注销此段，则输入变量优先级更高。
    globals_old = set(ds)
    globals_now = set(globals())
    need_add = globals_old-globals_now
    data = {k:ds[k] for k in need_add}
    ds = data
    
    
    # 对需要的操作（run）进行整理
    runs = []
    for arg in args:
        try:
           # 跳过unhashable type, e.g. list、set
           arg in {}
        except:
           pass
        else:
            # 自定义属性处理
            add_attr = {k:v for k,v in kwargs.items() if f'src.{k}' in arg}
            # n = 0
            while ((arg in kwargs) or add_attr):
                try:
                    
                    arg = kwargs[arg]
                    add_attr = {k:v for k,v in kwargs.items() if f'src.{k}' in arg}
                except:
                    # 需要函数表达式操作的属性
                    for attr,attr_run in add_attr.items(): 
                        
                        ## 如果存在这个属性且不修改(not amend)则跳过
                        # if hasattr(src, attr) & (not(amend)):
                        #     continue
                        
                        # 调用getattrs获得属性
                        attr_value = getattrs(src,attr_run,ds=ds,**kwargs)
                        
                        # 为源数据添加属性------1/2
                        setattr(src,attr,attr_value)
                        
                        ## 不会改变源数据，而是更新表达式及可用变量集(ds)------1/2
                        # arg = arg.replace(attr,f'__attr{n}')
                        # ds.update({f'__attr{n}':attr_value})
                        # n += 1
                    break
        

        if isiterable(arg) & (type(arg) != str):
            # 递归
            run = getattrs(src, *arg, get_runs=True, **kwargs)
        else:
            # 参数处理
            if (r'//ks//' in arg):
                arg,ks = arg.split(r'//ks//')
                
                ks = eval(ks)  #<<<<<<取出默认参数
                ks.update(ds)  #<<<<<<输入参数覆盖默认参数
                ds = ks 
            
            
            # 整理run
            # 为源数据添加属性------2/2
            run = f'src.{arg}' if not ('src' in arg) else arg
            
            ## 不会改变源数据，而是更新表达式及可用变量集(ds)------2/2
            # run = f'src.{arg}' if (not ('src' in arg)) & (n==0) else arg
            
        runs.append(run)
    # 返回runs
    if get_runs:
        return runs
    # 调用批量操作函数
    else:
        return evals(*runs,**ds)


def add_attrs(src, run=False, ds={}, **attrs_dict):
    """
    向类（src）中添加属性

    Parameters
    ----------
    src : 输入类
    run : 是否启用表达式操作函数.  The default is False.
    ds : TYPE, optional
        表达式所需变量，为空则只有此文件全局变量可用. The default is {}.
    
    attrs_dict : (dict)
        属性：   对应操作表达式(run为True)
              or        对应值(run为False)

    Returns
    -------
    None.

    """
    
    
    for attr,attr_run in attrs_dict.items():
        if run:
            value = getattrs(src,attr_run,ds=ds,**attrs_dict)
        else:
            value = attr_run
        
        setattr(src,attr,value)





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







    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        





































