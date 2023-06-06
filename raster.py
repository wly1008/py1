# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 14:59:27 2023

@author: wly
"""



from functools import partial
import rasterio
import pandas as pd
import numpy as np
from rasterio.enums import Resampling




def get(src,attrs):
    
    name = locals()
    
    returns = []
    for attr in attrs:
        name[attr] = getattr(src,attr)
        returns.append(name[attr])
    if len(returns) == 1:
        returns = returns[0]
    
    return returns
    

def read(path_tif, n=1, tran=True, nan=np.nan, dtype=np.float64, 
         re_shape=False, re_scale=False, re_size=False, how='nearest',printf=False):
    '''

    path : string
        地址.
    n : 1 or 2 or 3, optional.
        返回几个值. The default is 1.
    tran : bool, optional.
        是否变为单列. The default is True.
    nan : optional
        无效值设置.The default is np.nan.
    dtype : optional
        矩阵值的格式. The default is np.float64.
    
    ---------------------------------------------------------------------------
    单波段栅格
    
    重采样参数(re_shape=False, re_scale=False, re_len=False, how='nearest',printf=False)
    
    
    
    re_shape:形状重采样
    (count, height, width)
    
    re_size:大小重采样
    (xsize,ysize)
    
    re_scale:倍数重采样
    scale = 目标边长大小/原数据边长大小
    
    how:重采样方法(str or int)
    默认'nearest'
    
    (部分)
    mode:众数，6;
    nearest:临近值，0;
    bilinear:双线性，1;
    cubic_spline:三次卷积，3。
    
    printf : 任意值,optional.
        如果发生重采样，则会打印原形状及输入值。The default is False.
    
    --------------------------------------------------------------------------- 
    
    
    Returns:
        栅格矩阵（单列or原型）；profile;shape

    '''

    def updata():  #<<<<<<<<<更新函数
    
        if shape != out_shape:
            
            if printf:
                print(shape, printf)
            
            bounds = {'west': west, 'south': south, 'east': east,
                      'north': north, 'height': out_shape[1], 'width': out_shape[2]}

            transform = rasterio.transform.from_bounds(**bounds)

            profile.data.update({'height': out_shape[1], 'width': out_shape[2], 'transform': transform})
            
            if type(how) is int:
                resampling = how
            else:
                resampling = getattr(Resampling, how)
            
            
            data = src.read(out_shape=out_shape, resampling=resampling).astype(dtype)
        else:
            data = src.read().astype(dtype)
            pass    
        return data
    
    
    src = rasterio.open(path_tif)
    
    # 取出所需参数
    nodata, profile, count, height, width,transform = get(src,('nodata', 'profile', 'count', 'height', 'width','transform'))
    
    west, south, east, north = rasterio.transform.array_bounds(height, width, transform)
    shape = (count, height, width)
    nodata = dtype(nodata)
    
    # 重采样
    if re_shape:
        out_shape = re_shape
        
        # 更新矩阵、profile、shape
        data = updata()
        shape = out_shape
    
    
    elif re_size:
        xsize,ysize = re_size
        out_shape = (count,int((north-south)/ysize),int((east-west)/xsize))
        
        # 更新
        data = updata()
        shape = out_shape
    
    
    elif re_scale:
        scale = re_scale
        out_shape = (count, int(height/scale), int(width/scale))
        
        # 更新
        data = updata()
        shape = out_shape
    

    else:
        data = src.read().astype(dtype)  #<<<<<<矩阵
    
    # 变形
    if tran:
        data = data.reshape(-1, 1)
    else:
        data = data[0]
    data = pd.DataFrame(data)
    data.replace(nodata, nan, inplace=True)
    
    src.close()
    # 返回
    if n == 1:
        return data
    elif n == 2:
        return data,profile
    elif n == 3:
        return data,profile,shape
    else:
        print('n=1 or 2 or 3')


def out(tif_path, df, shape, pro):
    '''
    操作函数
    ---------

    输出文件函数

    '''
    df = pd.DataFrame(np.array(df).reshape(shape)[0])

    with rasterio.open(tif_path, 'w', **pro) as src:
        src.write(df, 1)


def mask(path_tif, path_mask, tif_path):
    '''
    操作函数，会直接输出
    ---------------------

    需保证meta一致才能正常使用

    -----------------------------------

    栅格提取栅格，在mask中有效值的位置，tif的相应位置值会被保留


    '''

    df_tif, pro, shape = read(path_tif, 3)
    df_mask, pro_m, shape_m = read(path_mask, 3)

    if pro != pro_m:
        print('meta不同')
        return
    mask = ~df_mask.isna()

    df = df_tif[mask]

    out(tif_path, df, shape, pro)
















































