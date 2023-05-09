# -*- coding: utf-8 -*-
"""
Created on Wed May  3 10:32:33 2023

@author: wly
"""

import rasterio
import pandas as pd
import numpy as np
from rasterio.enums import Resampling


def func(path,n = 1,tran = True, nan = np.nan, resampling = (False, 'bilinear'), print_ = False):
    '''
   
    
    ----------
    path : string
        地址.
    n : 1 or 2 or 3, optional.
        返回几个值. The default is 1.
    tran : bool, optional.
        是否变为单列. The default is True.
        
    resampling : tuple,optional.
        重采样.The default is (False, 'bilinear').
        (shape_，how)
        ---shape_---
        (src.count, src.height, src.width)
        ---how---
        (部分)
        mode:众数，6;
        nearest:临近值，0;
        bilinear:双线性，1;
        cubic_spline:三次卷积，3。
    Returns:
        栅格矩阵（单列or原型）；profile;shape

    '''
    
    with rasterio.open(path) as src:
        nodata = np.float32(src.nodata)
        profile = src.profile
        profile.data['dtype'] = 'float32'
        
        shape = (src.count, src.height, src.width)
        
        #重采样

        if resampling[0]:
            
            shape_ = resampling[0]
            if shape != shape_:
                
                if not print_ is False:
                    print(shape,print_)
                    
                shape = shape_
                # 重采样方法---how
                if type(resampling[1]) is int:
                    how = resampling[1]
                else:
                    how = getattr(Resampling,resampling[1])
            
                data = src.read(out_shape=shape_,resampling=how).astype('float32')
            else:
                resampling = (False,)
                
        if resampling[0] is False:
            data = src.read().astype('float32')
        data[data == nodata] = nan
        # 变形
        if tran:
            data = data.reshape(-1,1)
        else:
            data = data[0]
        data = pd.DataFrame(data)
    # 返回
    if n == 1:
        return data
    elif n == 2:
        return data,profile
    elif n == 3:
        return data,profile,shape
    else:
        print('n=1 or 2 or 3')
        
def out(path,df,shape,pro):
    '''
    输出文件函数

    '''
    df = pd.DataFrame(np.array(df).reshape(shape)[0])  
    
    with rasterio.open(path, 'w', **pro) as src:
        src.write(df, 1) 




def get_shape(path):
    with rasterio.open(path) as src:
        shape = src.shape
    return shape























