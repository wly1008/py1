# -*- coding: utf-8 -*-
"""
Created on Sat 2023/6/19 19:42
@Author : wly
"""

from rasterio.windows import Window

import rasterio.mask
import mycode.code as cd
import re
from functools import partial
import rasterio
import pandas as pd
import numpy as np
from rasterio.warp import calculate_default_transform
from rasterio.enums import Resampling
import os, sys
import inspect


def create_raster(**kwargs):
    memfile = rasterio.MemoryFile()
    return memfile.open(**kwargs)


def get_RasterArrt(raster_in, *args, ds={}, **kwargs):
    """
    获得栅格数据属性

    raster_in : (str or io.DatasetReader or io.DatasetWriter...(in io.py))
        栅格数据或栅格地址
    args: 所需属性或函数（类中存在的，输入属性名、函数名即可）
    ds: （dict）传递操作所需变量,可将全局变量（globals()先赋予一个变量，直接将globals()填入参数可能会报错）输入，默认变量为此文件及cd文件的全局变量


    kwargs: 字典值获得对应属性所需操作，可为表达式，默认参数以字典形式写在“//ks//”之后，在ds中输入相应变量可替代默认参数
            非自身类函数调用时及自身在dic、kwargs中定义的属性调用时，src不可省略。
            必须使用src代表源数据。

            合并属性返回类型为list. e.g.'raster_size': ('height', 'width') -> [900, 600]
            如需特定属性请用函数. e.g. 'raster_size': r"(src.height, src.width)" or r"pd.Serise([src.height, src.width])"
   （dic中有部分，按需求添加，可直接修改dic,效果一致,getattrs中ds参数是为传递操作所需变量,如在dic中添加ds需考虑修改函数参数名及系列变动）

    ---------------------------------
    return:
        args对应属性值列表


    """

    ## 输入变量优先级高
    # now = globals()
    # now.update(ds)
    # ds = now

    # 此文件变量优先级高
    ds.update(globals())

    dic = {'raster_size': r"(src.height, src.width)", 'cell_size': ('xsize', 'ysize'),
           'bends': 'count', 'xsize': r'transform[0]', 'ysize': r'abs(src.transform[4])',
           'values': r'src.read().astype(dtype)//ks//{"dtype":np.float64}',
           'arr':r'src.values.reshape(-1, 1)',
           'df': r'pd.DataFrame(src.values.reshape(-1, 1))',
           'shape_b':('count', 'height', 'width')}
    _getattrs = partial(cd.getattrs, **dic)

    src = raster_in if type(raster_in) in (i[1] for i in inspect.getmembers(rasterio.io)) else rasterio.open(raster_in)


    return _getattrs(src, *args, ds=ds, **kwargs)


def add_attrs_raster(src, ds={}, **kwargs):
    """
    向栅格数据中添加属性

    src:栅格数据
    ds:表达式所需变量
    kwargs:属性：对应表达式（"//ks//后为默认参数，在ds中输入相应变量可替代默认参数"）

    """
    dic = {'raster_size': r"(src.height, src.width)", 'cell_size': ('xsize', 'ysize'),
           'bends': 'count', 'xsize': r'transform[0]', 'ysize': r'abs(src.transform[4])',
           'values': r'src.read().astype(dtype)//ks//{"dtype":np.float64}',
           'arr':r'src.values.reshape(-1, 1)',
           'df': r'pd.DataFrame(src.values.reshape(-1, 1))',
           'shape_b':('count', 'height', 'width')}

    dic.update(kwargs)

    data = globals()
    data.update(ds)
    ds = data

    cd.add_attrs(src, run=True, ds=ds, **dic)


def window(raster_in,shape):
    '''
    

    Parameters
    ----------
    raster_in : (str or io.DatasetReader or io.DatasetWriter...(in io.py))
        栅格数据或栅格地址
    shape : tuple
          (height,width)
        分割为 height*width个窗口

    Returns
    -------
    windows : TYPE
        窗口集
    inxs : TYPE
        对应窗口在栅格中的位置索引

    '''
    
    
    
    
    src = raster_in if type(raster_in) in (i[1] for i in inspect.getmembers(rasterio.io)) else rasterio.open(raster_in)
    
    

    xsize,xend = src.width//shape[1], src.width%shape[1]
    ysize,yend = src.height//shape[0], src.height%shape[0]

    
    y_off = 0
    inxs = []
    inx = {}
    windows = []
    for ax0 in range(shape[0]):
        
        x_off = 0
        
        if (ax0 == (shape[0]-1)):
            height = ysize + yend
        else:
            height = ysize
        
        
        for ax1 in range(shape[1]):
            
            if (ax1 == (shape[1]-1)):
                width = xsize + xend
            else:
                width = xsize
            
            
            windown = Window(x_off,y_off,width,height)
            
            windows.append(windown)
            
            
            #------------------------------
            start = x_off
            end = x_off + width
            inx['x']=(start,end)
            
            start = y_off
            end = y_off + height
            inx['y'] =(start,end)
            
            inxs.append(inx.copy())
            #------------------------------
            
            x_off += width
             
        y_off += height
    
    return windows,inxs
    

def read(raster_in,
         n=1, tran=True,get_df = True,
         nan=np.nan, dtype=np.float64):
    """
    

    Parameters
    ----------
    raster_in : (str or io.DatasetReader or io.DatasetWriter...(in io.py))
        栅格数据或栅格地址
    n : 1 or 2 or 3, optional.
        返回几个值. The default is 1.
    tran : bool, optional.
        是否变为单列. The default is True.
    get_df : bool, optional.
        是否变为DataFrame
    nan : optional
        无效值设置.The default is np.nan.
    dtype : 数据类型（class），optional
        矩阵值的格式. The default is np.float64.

    Returns
    -------
    
        栅格矩阵（单列or原型）；profile;shape

    """
    
    
    
    
    
    
    src = raster_in if type(raster_in) in (i[1] for i in inspect.getmembers(rasterio.io)) else rasterio.open(raster_in)
    arr = src.read().astype(dtype)
    nodata = dtype(src.nodata)
    shape = arr.shape
    profile = src.profile
    profile.update(nodata=nodata)
    
    profile.update({'dtype': dtype,
                    'nodate': nan})
    df = pd.DataFrame(arr.reshape(-1, 1))
    df.replace(nodata, nan, inplace=True)
    if tran:
        if get_df:
            data = df
        else:
            data = np.array(df)
   
    else:
        if (shape[0] == 1) & (get_df):
            data = pd.DataFrame(np.array(df).reshape(shape)[0])
            
        else:
            data = np.array(df).reshape(shape)
            

    # 返回
    if n == 1:
        return data
    elif n == 2:
        return data, profile
    elif n == 3:
        return data, profile, shape
    else:
        print('n=1 or 2 or 3')



def out(out_path, data, profile, shape=None):
    """
    
    输出函数，
    可配合形变矩阵（设置shape原形状参数）
    


    """
    
    if not(shape is None):
        if len(shape)==2:
            shape = [1] + [i for i in shape]
        data = np.array(data).reshape(shape)
    
    elif len(data.shape) == 2:
        shape = [1] + [i for i in data.shape]
        data = np.array(data).reshape(shape)
    
        
    with rasterio.open(out_path,'w',**profile) as src:
        src.write(data)
    


def out_ds(ds,out_path):
    """
    输出栅格数据

    Parameters
    ----------
    ds : 
        栅格数据
    out_path : str
        输出地址

    Returns
    -------
    无

    """
    
    arr = ds.read()
    profile = ds.profile
    with rasterio.open(out_path,'w',**profile) as src:
        src.write(arr)
    

def clip(raster_in, dst_in=None, bounds=None, out_path=None, get_ds=True):
    """
    栅格按范围裁剪
    
    

    Parameters
    ----------
    raster_in : (str or io.DatasetReader or io.DatasetWriter...(in io.py))
        输入栅格数据或栅格地址
    dst_in : (str or io.DatasetReader or io.DatasetWriter...(in io.py)), optional
        目标范围的栅格数据或栅格地址
    bounds : tuple, optional
        目标范围(左，下，右，上)
    out_path : str, optional
        输出地址. The default is None.
    get_ds : bool, optional
        返回裁剪后的栅格数据(io.DatasetWriter). The default is True.

    Raises
    ------
        dst_in 和 bounds必须输入其中一个
    
        

    Returns
    -------
    if out_path:生成栅格文件，不返回
    elif get_ds:返回栅格数据(io.DatasetWriter)
    else:返回裁剪后的栅格矩阵（array）和 profile

    """
    
    
    
    
    
    
    
    src = raster_in if type(raster_in) in (i[1] for i in inspect.getmembers(rasterio.io)) else rasterio.open(raster_in)

    if dst_in:
        bounds = get_RasterArrt(dst_in, 'bounds')
    elif bounds:
        pass
    else:
        raise "dst_in 和 bounds必须输入其中一个"

    xsize, ysize, bounds_src, profile = get_RasterArrt(src, 'xsize', 'ysize', 'bounds', 'profile')

    # 填充范围
    union = (min(bounds[0], bounds_src[0]),  # west
             min(bounds[1], bounds_src[1]),  # south
             max(bounds[2], bounds_src[2]),  # east
             max(bounds[3], bounds_src[3]),)  # north

    arr = src.read()
    
    
    
    union_shape = (src.count, int((union[3] - union[1]) / ysize)+1, int((union[2] - union[0]) / xsize)+1)
    a = int((bounds_src[0] - union[0]) / xsize)
    d = int((union[3] - bounds_src[3]) / ysize)
    union_arr = np.full(union_shape, src.nodata, profile['dtype'])
    union_arr[:, d:d + src.height, a:a + src.width] = arr

    # clip
    a = int((bounds[0] - union[0]) / xsize)
    b = int((union[3] - bounds[1]) / ysize)
    c = int((bounds[2] - union[0]) / xsize)
    d = int((union[3] - bounds[3]) / ysize)

    dst_height = b - d
    dst_width = c - a
    dst_arr = union_arr[:, d:b, a:c]

    dst_transform = rasterio.transform.from_bounds(*bounds, dst_width, dst_height)

    profile.update({'height': dst_height,
                    'width': dst_width,
                    'transform': dst_transform})
    if out_path:
        with rasterio.open(out_path, 'w', **profile) as ds:
            ds.write(dst_arr)
    elif get_ds:
        ds = create_raster(**profile)
        ds.write(dst_arr)
        return ds
    else:
        return dst_arr, profile


def resampling(raster_in, out_path=None, get_ds=True,
               re_shape=False, re_scale=False, re_size=False, how='nearest', printf=False):
    """
    栅格重采样



    Parameters
    ----------
    raster_in : (str or io.DatasetReader or io.DatasetWriter...(in io.py))
        输入栅格数据或栅格地址
    out_path : str, optional
        输出地址. The default is None.
    get_ds : TYPE, optional
        返回裁剪后的栅格数据(io.DatasetWriter). The default is True.
    
    
    
    ----------
    
    
    re_shape:形状重采样(tuple)
    (count, height, width)

    re_size:大小重采样(tuple or number)
    (xsize,ysize) or size

    re_scale:倍数重采样(number)
    scale = 目标边长大小/源数据边长大小


    how:(str or int) , optional.
    重采样方式，The default is nearest.

    (部分)
    mode:众数，6;
    nearest:临近值，0;
    bilinear:双线性，1;
    cubic_spline:三次卷积，3。
    ...其余见rasterio.enums.Resampling




    printf : 任意值,optional.
        如果发生重采样，则会打印原形状及输入值。The default is False.

    Returns
    -------
    if out_path:生成栅格文件，不返回
    elif get_ds:返回栅格数据(io.DatasetWriter)
    else:返回重采样后的栅格矩阵（array）和 profile

    """
    

    
    def update():  # <<<<<<<<<更新函数

        if shape != out_shape:

            if not (printf is False):
                print(f'{printf}的原形状为{shape}')

            bounds = {'west': west, 'south': south, 'east': east,
                      'north': north, 'height': out_shape[1], 'width': out_shape[2]}

            transform = rasterio.transform.from_bounds(**bounds)

            profile.data.update({'height': out_shape[1], 'width': out_shape[2], 'transform': transform})

            resampling_how = how if type(how) is int else getattr(Resampling, how)
            data = src.read(out_shape=out_shape, resampling=resampling_how)
        else:
            data = src.read()

        return data

    src = raster_in if type(raster_in) in (i[1] for i in inspect.getmembers(rasterio.io)) else rasterio.open(raster_in)

    # 取出所需参数
    nodata, profile, count, height, width, transform = get_RasterArrt(src, *(
        'nodata', 'profile', 'count', 'height', 'width', 'transform'))
    west, south, east, north = rasterio.transform.array_bounds(height, width, transform)
    shape = (count, height, width)

    if re_shape:
        if len(re_shape) == 2:
            re_shape = [count] + [i for i in re_shape]
        out_shape = re_shape

        # 更新
        data = update()
        shape = out_shape


    elif re_size:

        if (type(re_size) == int) | (type(re_size) == float):
            xsize = re_size
            ysize = re_size
        else:
            xsize, ysize = re_size
        out_shape = (count, int((north - south) / ysize), int((east - west) / xsize))

        # 更新
        data = update()
        shape = out_shape



    elif re_scale:
        scale = re_scale
        out_shape = (count, int(height / scale), int(width / scale))

        # 更新
        data = update()
        shape = out_shape
    else:
        data = src.read()

    if out_path:
        with rasterio.open(out_path, 'w', **profile) as ds:
            ds.write(data)
    elif get_ds:
        ds = create_raster(**profile)
        ds.write(data)
        return ds
    else:
        return data, profile


def reproject(raster_in, dst_in=None,
              out_path=None, get_ds=True,
              crs=None,
              how='nearest', 
              resolution=None, shape=(None, None, None)):
    """
    栅格重投影

    Parameters
    ----------
    raster_in : (str or io.DatasetReader or io.DatasetWriter...(in io.py))
        输入栅格数据或栅格地址
    dst_in : (str or io.DatasetReader or io.DatasetWriter...(in io.py)), optional
        目标投影的栅格数据或栅格地址
    out_path : str, optional
        输出地址. The default is None.
    get_ds : io.DatasetWriter, optional
        返回裁剪后的栅格数据(io.DatasetWriter). The default is True.

    crs : crs.CRS, optional
        目标投影. The default is None.
        
    
    
    how:(str or int) , optional.
    重采样方式，The default is nearest.

    (部分)
    mode:众数，6;
    nearest:临近值，0;
    bilinear:双线性，1;
    cubic_spline:三次卷积，3。
    ...其余见rasterio.enums.Resampling
    
    
    resolution : TYPE, optional
        输出栅格分辨率. The default is None.
    shape : TYPE, optional
        输出栅格形状. The default is (None, None, None).
        
        
   
    Raises
    ------
        dst_in 和 bounds必须输入其中一个
    
    Returns
    -------
    TYPE
        DESCRIPTION.

    """
     
    
    src = raster_in if type(raster_in) in (i[1] for i in inspect.getmembers(rasterio.io)) else rasterio.open(raster_in)
    if crs:
        pass
    elif dst_in:
        crs = get_RasterArrt(dst_in,'crs')
    else:
        raise "dst_in 和 bounds必须输入其中一个"

    profile = src.profile
    if len(shape) == 2:
        shape = [src.count] + [i for i in shape]

    dst_transform, dst_width, dst_height = calculate_default_transform(src.crs, crs, src.width, src.height, *src.bounds,
                                                                       resolution=resolution, dst_width=shape[2],
                                                                       dst_height=shape[1])

    out_shape = (src.count, dst_height, dst_width)

    profile.update({'crs': crs, 'transform': dst_transform, 'width': dst_width, 'height': dst_height})

    how = how if type(how) is int else getattr(Resampling, how)

    arr = src.read(out_shape=out_shape, resampling=how)

    if out_path:
        with rasterio.open(out_path, 'w', **profile) as ds:
            ds.write(arr)
    elif get_ds:
        ds = create_raster(**profile)
        ds.write(arr)
        return ds
    else:
        return arr, profile


def unify(raster_in, dst_in, out_path=None, get_ds=True):
    '''
    统一栅格数据(空间参考、范围、行列数)

    Parameters
    ----------
    raster_in : (str or io.DatasetReader or io.DatasetWriter...(in io.py))
        输入栅格数据或栅格地址
    dst_in : (str or io.DatasetReader or io.DatasetWriter...(in io.py)), optional
        目标投影的栅格数据或栅格地址
    out_path : str, optional
        输出地址. The default is None.
    get_ds : io.DatasetWriter, optional
        返回统一后的栅格数据(io.DatasetWriter). The default is True.

    Returns
    -------
    if out_path:生成栅格文件，不返回
    elif get_ds:返回栅格数据(io.DatasetWriter)
    else:返回重采样后的栅格矩阵（array）和 profile

    '''
    
    
    shape = get_RasterArrt(dst_in, 'shape')

    ds_pro = reproject(raster_in=raster_in, dst_in=dst_in)
    ds_clip = clip(raster_in=ds_pro, dst_in=dst_in)

    return resampling(raster_in=ds_clip, out_path=out_path, get_ds=get_ds, re_shape=shape)



if __name__ == '__main__':

    path_in = r'F:/PyCharm/pythonProject1/arcmap/010栅格数据统一/蒸散/2001.tif'


    dst_in = r'F:/PyCharm/pythonProject1/arcmap/010栅格数据统一/降水(目标数据)/2001.tif'


    out_path = r'F:\PyCharm\pythonProject1\arcmap\010栅格数据统一\new\测试_clip2.tif'



    ds = unify(path_in, dst_in,  out_path=None, get_ds=True)


    out_ds(ds, out_path)










