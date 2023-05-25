# -*- coding: utf-8 -*-
"""
Created on Wed May  3 10:32:33 2023

@author: wly
"""

import rasterio
import pandas as pd
import numpy as np
from rasterio.enums import Resampling


def read(path_tif,n = 1,tran = True, nan = np.nan, resampling = ('shape_', 'how'), print_ = False):
    '''
    -----------
    
    重采样未解决仿射变换问题，暂时无法直接使用，只能得到重采样后的矩阵，需配合正确仿射数据使用
    
    ----------
    path : string
        地址.
    n : 1 or 2 or 3, optional.
        返回几个值. The default is 1.
    tran : bool, optional.
        是否变为单列. The default is True.
    nan : optional
        无效值设置.The default is np.nan.
    resampling : tuple,optional.
        重采样.默认不开启，如需使用shape_、how都是必填的.
        (shape_，how)
        ---shape_---(tuple)
        (src.count, src.height, src.width)
        ---how---(str or int)
        (部分)
        mode:众数，6;
        nearest:临近值，0;
        bilinear:双线性，1;
        cubic_spline:三次卷积，3。
    print_ : 任意值,optional.
        如果发生重采样，则会打印原形状及输入值。The default is False.
    Returns:
        栅格矩阵（单列or原型）；meat;shape

    '''
    
    with rasterio.open(path_tif) as src:
        nodata = np.float64(src.nodata)
        # profile = src.profile
        # profile.data['dtype'] = 'float64'
        meat = src.meta
        meat['dtype'] = 'float64'
        shape = (src.count, src.height, src.width)
        
        #重采样
        if resampling[0] != 'shape_':
            
            shape_ = resampling[0]
            
            if shape != shape_:
                
                if not print_ is False:
                    print(shape,print_)
                    
                shape = shape_
                # profile.data.update({'height':shape_[1],'width':shape_[2]})
                
                meat.update({'height':shape_[1],'width':shape_[2]})
                
                # 重采样方法---how
                
                if type(resampling[1]) is int:
                    how = resampling[1]
                else:
                    how = getattr(Resampling,resampling[1])
            
                data = src.read(out_shape=shape_,resampling=how).astype('float64')
            else:
                resampling = ('shape_',)
            
        if resampling[0] == 'shape_':
            data = src.read().astype('float64')
        
        # 变形
        if tran:
            data = data.reshape(-1,1)
        else:
            data = data[0]
        data = pd.DataFrame(data)
        
        # 无效值处理
        data.replace(nodata,nan,inplace=True)
        # profile.data['nodata'] = nan
        meat['nodata'] = nan
    del src
    
    
    # 返回
    if n == 1:
        return data
    elif n == 2:
        # return data,profile
        return data,meat
    elif n == 3:
        # return data,profile,shape
        return data,meat,shape
    else:
        print('n=1 or 2 or 3')
 




def out(tif_path,df,shape,meat):
    '''
    操作函数
    ---------
    
    输出文件函数

    '''
    df = pd.DataFrame(np.array(df).reshape(shape)[0])  
    
    with rasterio.open(tif_path, 'w', **meat) as src:
        src.write(df, 1) 




def get_shape(path):
    with rasterio.open(path) as src:
        shape = src.shape
    return shape



def mask(path_tif, path_mask, tif_path):
    '''
    操作函数，会直接输出
    ---------------------
    
    需保证meat一致才能正常使用
    
    -----------------------------------
    
    栅格提取栅格，在mask中有效值的位置，tif的相应位置值会被保留


    '''
    
    
    df_tif, meat, shape = read(path_tif,3)
    df_mask, meat_m, shape_m = read(path_mask,3)

    if meat != meat_m:
        print('meat不同')
        return 
    mask = ~df_mask.isna()

    df = df_tif[mask]

    out(tif_path, df, shape, meat)




















