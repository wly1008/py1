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




def get(src,*attrs):
    
    name = locals()
    
    returns = []
    for attr in attrs:
        name[attr] = getattr(src,attr)
        returns.append(name[attr])
    if len(returns) == 1:
        returns = returns[0]
    
    return returns
    




def get_RasterArrt(path_in,*attrs):
    
    with rasterio.open(path_in) as src:
        return get(src,*attrs)



def read(path_in, n=1, tran=True, nan=np.nan, dtype=np.float64, 
         re_shape=False, re_scale=False, re_size=False, how='nearest', printf=False):
    '''

    path : string
        地址.
    n : 1 or 2 or 3, optional.
        返回几个值. The default is 1.
    tran : bool, optional.
        是否变为单列. The default is True.
    nan : optional
        无效值设置.The default is np.nan.
    dtype : 数据类型转换函数，optional
        矩阵值的格式. The default is np.float64.
    
    ---------------------------------------------------------------------------
    
    
    重采样参数（re_shape=False, re_scale=False, re_size=False, how='nearest',printf=False）
    
    
    
    re_shape:形状重采样
    (count, height, width)
    
    re_size:大小重采样
    (xsize,ysize)
    
    re_scale:倍数重采样
    scale = 目标边长大小/原数据边长大小
    
    ---how---(str or int)
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

    def update():  #<<<<<<<<<更新函数
    
        if shape != shape_aim:
            
            if printf:
                print(shape, printf)
            
            bounds = {'west': west, 'south': south, 'east': east,
                      'north': north, 'height': shape_aim[1], 'width': shape_aim[2]}

            transform = rasterio.transform.from_bounds(**bounds)

            profile.data.update({'height': shape_aim[1], 'width': shape_aim[2], 'transform': transform})
            
            if type(how) is int:
                _resampling = how
            else:
                _resampling = getattr(Resampling, how)
            
            
            data = src.read(out_shape=shape_aim, resampling=_resampling).astype(dtype)
        else:
            data = src.read().astype(dtype)
            pass    
        return data
    
    
    src = rasterio.open(path_in)
    
    # 取出所需参数
    nodata, profile, count, height, width,transform = get(src,*('nodata', 'profile', 'count', 'height', 'width','transform'))
    
    west, south, east, north = rasterio.transform.array_bounds(height, width, transform)
    shape = (count, height, width)
    nodata = dtype(nodata)
    
    
    if re_shape:
        shape_aim = re_shape
        
        # 更新矩阵、profile、shape
        data = update()
        shape = shape_aim
    
    
    elif re_size:
        xsize,ysize = re_size
        shape_aim = (count,int((north-south)/ysize),int((east-west)/xsize))
        
        # 更新
        data = update()
        shape = shape_aim
    
    
    
    elif re_scale:
        scale = re_scale
        shape_aim = (count, int(height/scale), int(width/scale))
        
        # 更新
        data = update()
        shape = shape_aim
    

    else:
        data = src.read().astype(dtype)
    
    
    # 处理无效值
    data = data.reshape(-1, 1)
    data = pd.DataFrame(data)
    data.replace(nodata, nan, inplace=True)
    
    # 变形
    if tran:
        pass
    else:
        data = np.array(data).reshape(shape)
        if shape[0] == 1:
            data = pd.DataFrame(data[0])
    
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


def out(out_path, data, shape, pro):
    '''
    操作函数
    ---------

    输出文件函数

    '''
    data = np.array(data).reshape(shape)

    with rasterio.open(out_path, 'w', **pro) as src:
        src.write(data)


def mask(path_in, path_mask, out_path):
    '''
    操作函数，会直接输出
    ---------------------

    需保证meta一致才能正常使用

    -----------------------------------

    栅格提取栅格，在掩膜（mask）中有效值的位置，输入栅格（in）的相应位置值会被保留


    '''

    df_tif, pro, shape = read(path_in, 3)
    df_mask, pro_m, shape_m = read(path_mask, 3)
    
    
    meta = get_RasterArrt(path_in,'meta')
    meta_m = get_RasterArrt(path_mask,'meta')
    
    
    
    if meta != meta_m:
        print('meat不同')
        return
    mask = ~df_mask.isna()

    df = df_tif[mask]

    out(out_path, df, shape, pro)





def resampling(path_in, out_path, nan=np.nan, dtype=np.float64, 
               re_shape=False, re_scale=False, re_size=False, how='nearest', printf=False):
    
    
    '''
    操作函数，直接输出
    ------------------------------------
    
    重采样，参数详见read、out
    
    '''

    data, pro, shape = read(path_in, nan=nan, dtype=dtype,
                            re_size=re_size, re_scale=re_scale, re_shape=re_shape,
                            how=how, printf=printf)
    
    
    out(out_path, data, shape, pro)
    
    




































