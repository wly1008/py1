# -*- coding: utf-8 -*-
"""
Created on Sat 2023/6/19 19:42
@Author : wly
"""

from rasterio.windows import Window
import warnings
import rasterio.mask
import mycode.codes as cd
from functools import partial
import rasterio
import pandas as pd
import numpy as np
from rasterio.warp import calculate_default_transform
from rasterio.enums import Resampling
import os, sys, re
import inspect
from rasterio.warp import reproject as _reproject


def create_raster(**kwargs):
    memfile = rasterio.MemoryFile()
    return memfile.open(**kwargs)



def get_RasterArrt(raster_in, *args, ds={}, **kwargs):
    """
    获得栅格数据属性

    raster_in : (str or io.DatasetReader or io.DatasetWriter...(in io.py))
        栅格数据或栅格地址
    *args: 所需属性或函数（类中存在的，输入属性名、函数名即可）
    ds: (dict)
        传递操作所需变量,可将全局变量(globals()先赋予一个变量，直接将globals()填入参数可能会报错)输入，
        默认额外可用变量为函数此文件及mycode.code文件的全局变量


    **kwargs: 字典值获得对应属性所需操作，可为表达式，默认参数以字典形式写在“//ks//”之后，在ds中输入相应变量可替代默认参数
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

    dic = {'raster_size': r"(src.height, src.width)", 'cell_size': r"(src.xsize, src.ysize)",
           'bends': 'count', 'xsize': r'transform[0]', 'ysize': r'abs(src.transform[4])',
           'Bounds': r'[float(f"{i:f}") for i in src.bounds]',
           'values': r'src.read().astype(dtype)//ks//{"dtype":np.float64}',
           'arr': r'src.values.reshape(-1, 1)',
           'df': r'pd.DataFrame(src.values.reshape(-1, 1))',
           #'shape_b': r"(src.count, src.height, src.width)"}
           'shape_b': ('count', 'height', 'width')}
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
    dic = {'raster_size': r"(src.height, src.width)", 'cell_size': r"(src.xsize, src.ysize)",
           'bends': 'count', 'xsize': r'transform[0]', 'ysize': r'abs(src.transform[4])',
           'Bounds': r'[float(f"{i:f}") for i in src.bounds]',
           'values': r'src.read().astype(dtype)//ks//{"dtype":np.float64}',
           'arr': r'src.values.reshape(-1, 1)',
           'df': r'pd.DataFrame(src.values.reshape(-1, 1))',
           #'shape_b': r"(src.count, src.height, src.width)"}
           'shape_b': ('count', 'height', 'width')}

    dic.update(kwargs)

    data = globals()
    data.update(ds)
    ds = data

    cd.add_attrs(src, run=True, ds=ds, **dic)



def check(raster_in, dst_in, need=None, *args):
    '''
    检验栅格数据是否统一
    (空间参考、范围、栅格行列数)
    (Bounds为bounds精确到小数点后六位)

    Parameters
    ----------
    输入两栅格
    
    need : 完全自定义比较属性
    *args : 添加其他需要比较的属性
    
    raster_in : TYPE
        DESCRIPTION.
    dst_in : TYPE
        DESCRIPTION.

    Returns
    -------
    True or False,
    不同属性的列表
    

    '''
    if need:
        arrtnames = need
    else:
        arrtnames = ['crs', 'Bounds', 'raster_size'] + [i for i in args if not(i in ['crs', 'Bounds', 'raster_size'])]
    src_arrts = get_RasterArrt(raster_in, arrtnames)
    dst_arrts = get_RasterArrt(dst_in, arrtnames)
    
    
    
    
    
    diffe = [arrtnames[i] for i in range(len(arrtnames)) if src_arrts[i] != dst_arrts[i]]
    
    if diffe == []:
        return True,[]
    else:
        return False,diffe
    
    

def _return(out_path,get_ds,arr,profile):
    if out_path:
        with rasterio.open(out_path, 'w', **profile) as ds:
            ds.write(arr)
    elif get_ds:
        ds = create_raster(**profile)
        ds.write(arr)
        return ds
    else:
        return arr,profile



def window(raster_in, shape):
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

    xsize, xend = src.width // shape[1], src.width % shape[1]
    ysize, yend = src.height // shape[0], src.height % shape[0]
    
    
    y_off = 0
    y_inx = 0
    inxs = []
    # inx = {}
    windows = []
    for ax0 in range(shape[0]):
        
        
        x_off = 0
        x_inx = 0

        if (ax0 == (shape[0] - 1)):
            height = ysize + yend
        else:
            height = ysize

        for ax1 in range(shape[1]):

            if (ax1 == (shape[1] - 1)):
                width = xsize + xend
            else:
                width = xsize

            windown = Window(x_off, y_off, width, height)

            windows.append(windown)
            
            
            
            '''
            
            start = x_off
            end = x_off + width
            inx['x'] = (start, end)

            start = y_off
            end = y_off + height
            inx['y'] = (start, end)

            inxs.append(inx.copy())
            '''
            
            inxs.append((y_inx,x_inx))
            
            x_off += width
            x_inx += 1
        
        y_off += height
        y_inx += 1

    return windows, inxs


def read(raster_in,
         n=1, tran=True, get_df=True,
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
        是否变为DataFrame，The default is True.
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
    profile.update({'dtype': dtype,
                    'nodate': nan})
    
    # 变形，无效值处理
    df = pd.DataFrame(arr.reshape(-1, 1))
    df.replace(nodata, nan, inplace=True)
    
    # 是否保留变形，是否变为df
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


def out(out_path, data, profile):
    """
    
    输出函数，
    可配合形变矩阵
    


    """
    
    shape = (profile['count'], profile['height'], profile['width'])
    
    if data.shape != shape:
        data = np.array(data).reshape(shape)

    with rasterio.open(out_path, 'w', **profile) as src:
        src.write(data)


def out_ds(ds, out_path):
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
    with rasterio.open(out_path, 'w', **profile) as src:
        src.write(arr)


def resampling(raster_in, out_path=None, get_ds=True,
               re_shape=None, re_scale=None, re_size=None, how='mode', printf=False):
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
    重采样类型三选一，都不输入则原样返回
    
    re_shape:形状重采样(tuple)
    (height, width) or (count, height, width)

    re_size:大小重采样(tuple or number)
    (xsize,ysize) or size

    re_scale:倍数重采样(number)
    scale = 目标边长大小/源数据边长大小


    how:(str or int) , optional.
    重采样方式，The default is mode.

    (部分)
    mode:众数，6;
    nearest:临近值，0;
    bilinear:双线性，1;
    cubic_spline:三次卷积，3。
    ...其余见rasterio.enums.Resampling

    printf : 任意值,optional.
        如果发生重采样，则会打印原形状及输入的printf值。The default is False.

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

            transform = rasterio.transform.from_bounds(*bounds, height=out_shape[1], width=out_shape[2])

            profile.data.update({'height': out_shape[1], 'width': out_shape[2], 'transform': transform})

            resampling_how = how if isinstance(how, int) else getattr(Resampling, how)
            data = src.read(out_shape=out_shape, resampling=resampling_how)
        else:
            data = src.read()

        return data

    src = raster_in if type(raster_in) in (i[1] for i in inspect.getmembers(rasterio.io)) else rasterio.open(raster_in)

    # 取出所需参数
    nodata, profile, count, height, width, bounds= get_RasterArrt(src, *(
        'nodata', 'profile', 'count', 'height', 'width', 'bounds'))
    west, south, east, north = bounds
    shape = (count, height, width)

    if re_shape:
        if not(2 <= len(re_shape) <= 3):
            mis = 'resampling:\n当前函数接收re_shape=%s\n请输入二维("height", "width")或三维("count", "height", "width")形状' % str(re_shape)
            raise Exception(mis)
        
        re_shape = list(re_shape)
        re_shape = [count] + re_shape[-2:]

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
              how='mode',
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
    重采样方式，The default is mode.

    (部分)
    mode:众数，6;
    nearest:临近值，0;
    bilinear:双线性，1;
    cubic_spline:三次卷积，3。
    ...其余见rasterio.enums.Resampling
    
    ------------------------------------------------------
    resolution : TYPE, optional
        输出栅格分辨率单位为目标坐标单位. The default is None.
    shape : TYPE, optional
        输出栅格形状. The default is (None, None, None).
        
        ------<resolution与shape不能一起使用>
        
        
   
    Raises
    ------
        dst_in 和 bounds必须输入其中一个
    
    Returns
    -------
    if out_path:生成栅格文件，不返回
    elif get_ds:返回栅格数据(io.DatasetWriter)
    else:返回重投影后的栅格矩阵（array）和 profile

    """

    src = raster_in if type(raster_in) in (i[1] for i in inspect.getmembers(rasterio.io)) else rasterio.open(raster_in)
    if crs:
        pass
    elif dst_in:
        crs = get_RasterArrt(dst_in, 'crs')
    else:
        raise Exception("dst_in 和 bounds必须输入其中一个")

    profile = src.profile
    if len(shape) == 2:
        shape = [src.count] + list(shape)

    dst_transform, dst_width, dst_height = calculate_default_transform(src.crs, crs, src.width, src.height, *src.bounds,
                                                                       resolution=resolution, dst_width=shape[2],
                                                                       dst_height=shape[1])

    profile.update({'crs': crs, 'transform': dst_transform, 'width': dst_width, 'height': dst_height})

    how = how if isinstance(how, int) else getattr(Resampling, how)

    lst = []
    for i in range(1, src.count + 1):
        arrn = src.read(i)
        dst_array = np.empty((dst_height, dst_width), dtype=profile['dtype'])
        _reproject(  
            # 源文件参数
            source=arrn,
            src_crs=src.crs,
            src_transform=src.transform,
            # 目标文件参数
            destination=dst_array,
            dst_transform=dst_transform,
            dst_crs=crs,
            dst_nodata=src.nodata,
            # 其它配置
            resampling=how,
            num_threads=2)

        lst.append(dst_array)
    dst_arr = np.array(lst)

    if out_path:
        with rasterio.open(out_path, 'w', **profile) as ds:
            ds.write(dst_arr)
    elif get_ds:
        ds = create_raster(**profile)
        ds.write(dst_arr)
        return ds
    else:
        return dst_arr, profile

def extract(raster_in, dst_in,
            out_path=None, get_ds=True):
    """

    栅格按栅格掩膜提取
    (对掩膜栅格有效值位置栅格值进行提取)


    Parameters
    ----------
    raster_in : (str or io.DatasetReader or io.DatasetWriter...(in io.py))
        输入栅格数据或栅格地址
    dst_in : (str or io.DatasetReader or io.DatasetWriter...(in io.py)), optional
        掩膜的栅格数据或栅格地址
    out_path : str, optional
        输出地址. The default is None.
    get_ds : bool, optional
        返回提取后的栅格数据(io.DatasetWriter). The default is True.


    Raises
    ------
    Exception
        二者的'crs'、 'raster_size'、 'Bounds'需统一,可先调用unify函数统一栅格数据
        或使用mask函数



    Returns
    -------
    if out_path:生成栅格文件，不返回
    elif get_ds:返回提取后的栅格数据(io.DatasetWriter)
    else:返回提取后的栅格矩阵（array）和 profile


    """

    src = raster_in if type(raster_in) in (i[1] for i in inspect.getmembers(rasterio.io)) else rasterio.open(raster_in)
    dst = dst_in if type(dst_in) in (i[1] for i in inspect.getmembers(rasterio.io)) else rasterio.open(dst_in)

    # arrtnames = ('crs', 'raster_size', 'Bounds')

    # src_arrts = get_RasterArrt(src, arrtnames)
    # dst_arrts = get_RasterArrt(dst, arrtnames)
    
    judge,dif = check(src, dst)

    if not judge:
        
        mis = '\nextract 无法正确提取:\n'
        for i in dif:
            mis += f'\n    \"{i}\" 不一致'
        mis += '\n\n----<请统一以上属性>'
        raise Exception(mis)

    # 获得有效值掩膜
    mask_arr = dst.dataset_mask()

    if len(mask_arr.shape) == 3:
        mask_arr = mask_arr.max(axis=0)

    mask_arr = np.array([mask_arr for i in range(src.count)])

    # 按掩膜提取
    profile = src.profile
    nodata = src.nodata
    
    # uint8格式，None无法输出
    if (profile['dtype'] == 'uint8') & (nodata == None) :
        profile.update({'dtype':np.float64,'nodata': np.nan})
        nodata = np.nan
    arr = src.read()
    arr = np.where(mask_arr == 0, nodata, arr)

    

    if out_path:
        with rasterio.open(out_path, 'w', **profile) as ds:
            ds.write(arr)
    elif get_ds:
        ds = create_raster(**profile)
        ds.write(arr)
        return ds
    else:
        return arr, profile





def clip(raster_in,
         dst_in=None, bounds=None,
         Extract=False,
         out_path=None, get_ds=True):
    """
    栅格按范围裁剪
    (须保证投影一致)
    (可按栅格有效值位置掩膜提取)
    
    

    Parameters
    ----------
    raster_in : (str or io.DatasetReader or io.DatasetWriter...(in io.py))
        输入栅格数据或栅格地址
    dst_in : (str or io.DatasetReader or io.DatasetWriter...(in io.py)), optional
        目标范围的栅格数据或栅格地址
    bounds : tuple, optional
        目标范围(左，下，右，上)
    
    Extract : bool.optional
        调用extract函数
        是否对目标dst_in有效值位置的数据进行提取
        (类似矢量按周长边界裁剪栅格，dst_in必填且为栅格). The default is False.
    
    out_path : str, optional
        输出地址. The default is None.
    get_ds : bool, optional
        返回裁剪后的栅格数据(io.DatasetWriter). The default is True.

    Raises
    ------
       1. dst_in 和 bounds必须输入其中一个
       2. 如使用dst_in, dst_in 和 raster_in空间参考(crs)须一致.
       2. 使用extract，dst_in必填且为栅格
        

    Returns
    -------
    if out_path:生成栅格文件，不返回
    elif get_ds:返回栅格数据(io.DatasetWriter)
    else:返回裁剪后的栅格矩阵（array）和 profile

    """

    src = raster_in if type(raster_in) in (i[1] for i in inspect.getmembers(rasterio.io)) else rasterio.open(raster_in)

    if dst_in:

        bounds, crs = get_RasterArrt(dst_in, 'bounds', 'crs')
        if crs != src.crs:
            mis = '\nclip:\n \"crs\"不一致'
            raise Exception(mis)

    elif bounds:
        pass
    else:
        mis = "\nclip:\n\n    \"dst_in\"和\"bounds\"必须输入其中一个"
        raise Exception(mis)

    xsize, ysize, bounds_src, profile, nodata = get_RasterArrt(src, 'xsize', 'ysize', 'bounds', 'profile', 'nodata')

    # 判断是否有交集
    inter = (max(bounds[0], bounds_src[0]),  # west
             max(bounds[1], bounds_src[1]),  # south
             min(bounds[2], bounds_src[2]),  # east
             min(bounds[3], bounds_src[3]))  # north

    if (inter[2] <= inter[0]) | (inter[3] <= inter[1]):
        # print('输入范围与栅格不重叠')
        warnings.warn('\nclip: 输入范围与栅格不重叠')

    # 填充范围
    
    # uint8格式，None无法输出
    if (profile['dtype'] == 'uint8') & (nodata == None) :
         profile.update({'dtype':np.float64,'nodata': np.nan})
         nodata = np.nan

    
    

    # 并集
    union = (min(bounds[0], bounds_src[0]),  # west
             min(bounds[1], bounds_src[1]),  # south
             max(bounds[2], bounds_src[2]),  # east
             max(bounds[3], bounds_src[3]))  # north

    union_shape = (src.count, int((union[3] - union[1]) / ysize) + 1, int((union[2] - union[0]) / xsize) + 1)
    union_arr = np.full(union_shape, nodata, object)

    # 填入源数据栅格值
    arr = src.read()

    a = int((bounds_src[0] - union[0]) / xsize)
    d = int((union[3] - bounds_src[3]) / ysize)
    union_arr[:, d:d + src.height, a:a + src.width] = arr

    # clip,提取输入范围内的值
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

    if Extract:
        if dst_in is None:
            raise Exception('\nclip:\n    使用extract，dst_in必填且为栅格')

        # 如果栅格大小不同,重采样
        src_shape = (dst_height, dst_width)
        dst_shape = get_RasterArrt(dst_in, 'raster_size')

        if src_shape != dst_shape:
            dst = resampling(raster_in=dst_in, re_shape=src_shape)
        else:
            dst = dst_in

        ds = create_raster(**profile)
        ds.write(dst_arr)

        return extract(raster_in=ds, dst_in=dst, out_path=out_path, get_ds=get_ds)

    if out_path:
        with rasterio.open(out_path, 'w', **profile) as ds:
            ds.write(dst_arr)
    elif get_ds:
        ds = create_raster(**profile)
        ds.write(dst_arr)
        return ds
    else:
        return dst_arr, profile



def zonal(raster_in, dst_in, stats, dic=None):
    '''
    分区统计
    栅格统计栅格
    分区栅格应为整型栅格

    Parameters
    ----------
    raster_in : TYPE
        输入栅格
    dst_in : TYPE
        分区数据栅格
    stats : 
       统计类型。基于df.agg(stats) .e.g. 'mean' or ['mean','sum','max']...
    
    
    dic : dict
        分区数据栅格各值对应属性
    
    Raises
    ------
    Exception
        二者的'crs'、 'raster_size'、 'Bounds'需统一,可先调用unify函数统一栅格数据
        或使用zonal_u函数

    Returns
    -------
    所需统计值的dataframe

    '''
    
    src = raster_in if type(raster_in) in (i[1] for i in inspect.getmembers(rasterio.io)) else rasterio.open(raster_in)
    dst = dst_in if type(dst_in) in (i[1] for i in inspect.getmembers(rasterio.io)) else rasterio.open(dst_in)
    
    
    judge,dif = check(src, dst)

    if not judge:
        
        mis = '\nextract 无法正确提取:\n'
        for i in dif:
            mis += f'\n    \"{i}\" 不一致'
        mis += '\n\n----<请统一以上属性>'
        raise Exception(mis)
    
    
    df_src = read(src)
    df_dst = read(dst)
    
    df_return = pd.DataFrame(index=(['name']+stats))
    
    areas = list(df_dst[0].unique())
    
    if len(areas) >= 1000:
        warnings.warn('\n分区数为%d,分区栅格可能为浮点型栅格'%len(areas))
    
    
    
    for area in areas:
        
        serice = pd.Series(dtype='float64')
        try:
            serice['name'] = dic[area]
        except:
            serice['name'] = area

        value = df_src[df_dst.isin([area])].agg(stats,axis=0)  # isin()解决np.nan不被 == 检索问题
        serice = pd.concat([serice,value])
        df_return = pd.concat([df_return,serice],axis=1)
    return df_return.T



def unify(raster_in, dst_in,
          out_path=None, get_ds=True,
          Extract=False, how='mode',
          **kwargs):
    """
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
    Extract : bool, optional
        是否按有效值位置提取. The default is False
    
    how:(str or int) , optional.
    重采样方式，The default is mode.

    (部分)
    mode:众数，6;
    nearest:临近值，0;
    bilinear:双线性，1;
    cubic_spline:三次卷积，3。
    ...其余见rasterio.enums.Resampling
    


    **kwargs:
        接收调用函数(reproject、clip、resampling)的其他参数.
        e.g. printf(resampling中：如果发生重采样，则会打印原形状及printf值。The default is False.)

    Returns
    -------
    if out_path:生成栅格文件，不返回
    elif get_ds:返回栅格数据(io.DatasetWriter)
    else:返回重采样后的栅格矩阵（array）和 profile

    """
    
    
    
    # 是否按有效值位置提取
    if Extract:
        ds = unify(raster_in=raster_in, dst_in=dst_in, how=how, Extract=False, out_path=None, get_ds=True, **kwargs)
        kwargs_extract = {}  # 设置默认参数
        kwargs_extract.update({k: v for k, v in kwargs.items() if k in inspect.getfullargspec(extract)[0]}) #接收其他参数 
        
        return extract(raster_in=ds, dst_in=dst_in,out_path=out_path,get_ds=get_ds,**kwargs_extract)
    
    # 获得栅格变量
    src = raster_in if type(raster_in) in (i[1] for i in inspect.getmembers(rasterio.io)) else rasterio.open(raster_in)
    dst = dst_in if type(dst_in) in (i[1] for i in inspect.getmembers(rasterio.io)) else rasterio.open(dst_in)
    # 检查哪些属性需要统一
    judge,dif = check(src,dst)
    if judge:
        profile = src.profile
        arr = src.read()
        return _return(out_path,get_ds,arr,profile)

    elif 'crs' in dif:
        run = 3
    elif 'bounds' in dif:
        run = 2
    elif 'raster_size' in dif:
        run = 1
    else:
        raise Exception('有问题')
        

    
    shape = dst.shape
    
    
    ds = raster_in
    # 重投影（空间参考）
    if run == 3:
        kwargs_reproject = {}  # 设置默认参数
        kwargs_reproject.update({k: v for k, v in kwargs.items() if k in inspect.getfullargspec(reproject)[0]})  #接收其他参数 
        ds = reproject(raster_in=ds, dst_in=dst, how=how, **kwargs_reproject)
    # 裁剪（范围）
    
    if run >= 2:
        kwargs_clip = {}  # 设置默认参数
        kwargs_clip.update({k: v for k, v in kwargs.items() if k in inspect.getfullargspec(clip)[0]}) #接收其他参数 
        ds = clip(raster_in=ds, dst_in=dst,**kwargs_clip)
    # 重采样（行列数）
    kwargs_resapilg = {}  # 设置默认参数
    kwargs_resapilg.update({k: v for k, v in kwargs.items() if k in inspect.getfullargspec(resampling)[0]})  #接收其他参数 
    return resampling(raster_in=ds, out_path=out_path,how=how, get_ds=get_ds, re_shape=shape, re_size=False, re_scale=None, **kwargs_resapilg)
    
    
        
    
    


def clip_u(raster_in,dst_in=None,bounds=None,
         Extract=False,
         out_path=None, get_ds=True,**kwargs):
    '''
    栅格裁剪
    按范围裁剪。
    含临时统一操作，可提取。
    

    Parameters
    ----------
    raster_in : (str or io.DatasetReader or io.DatasetWriter...(in io.py))
       输入栅格数据或栅格地址
    dst_in : (str or io.DatasetReader or io.DatasetWriter...(in io.py)), optional
       目标范围的栅格数据或栅格地址
    bounds : tuple, optional
       目标范围(左，下，右，上)
   
    Extract : bool.optional
       调用extract函数
       是否对原栅格处于掩膜栅格有效值位置的值进行提取
       (类似矢量按周长边界裁剪栅格，dst_in必填且为栅格). The default is False.
   
    out_path : str, optional
       输出地址. The default is None.
    get_ds : bool, optional
       返回裁剪后的栅格数据(io.DatasetWriter). The default is True.
      
    Raises
    ------
       1. "dst_in"和"bounds"必须输入其中一个
       2. 使用extract，dst_in必填且为栅格
       
       
    Returns
    -------
    if out_path:生成栅格文件，不返回
    elif get_ds:返回栅格数据(io.DatasetWriter)
    else:返回裁剪后的栅格矩阵（array）和 profile
    



    '''


    # 按范围裁剪，无须统一
    if bounds:
        return clip(raster_in=raster_in, bounds=bounds, out_path=out_path, get_ds=get_ds)
    
    # 检查参数
    if not dst_in:
        excs = "\nclip:\n\n    \"dst_in\"和\"bounds\"必须输入其中一个"
        raise Exception(excs)
    
    # 保证空间参考统一
    src_crs = get_RasterArrt(raster_in,'crs')
    dst_crs = get_RasterArrt(dst_in,'crs')
    
    if src_crs != dst_crs:
        ds = reproject(dst_in,raster_in)
    
    # 调用clip函数
    return clip(raster_in=raster_in, dst_in=ds, Extract=Extract, out_path=out_path, get_ds=get_ds)
    
    
def mask(raster_in, dst_in,
         Clip=False,
         out_path=None, get_ds=True, **kwargs):
    """
    栅格按栅格掩膜提取,
    对原栅格处于掩膜栅格有效值位置的值进行提取，
    含临时统一操作

    --------------------------

    raster_in : (str or io.DatasetReader or io.DatasetWriter...(in io.py))
        输入栅格数据或栅格地址
    dst_in : (str or io.DatasetReader or io.DatasetWriter...(in io.py)), optional
        掩膜的栅格数据或栅格地址
    Clip : bool,optional
        是否裁剪
    out_path : str, optional
        输出地址. The default is None.
    get_ds : bool, optional
        返回提取后的栅格数据(io.DatasetWriter). The default is True.



    Returns
    ------------------
    if out_path:生成栅格文件，不返回
    elif get_ds:返回提取后栅格数据(io.DatasetWriter)
    else:返回提取后的栅格矩阵（array）和 profile

    """
    
    # 裁剪则调用clip_unify函数
    if Clip:
        return clip_u(raster_in, dst_in, Extract=True, out_path=out_path, get_ds=get_ds,**kwargs)
    
    # 先保证掩膜数据属性与输入数据相同
    ds = unify(raster_in=dst_in, dst_in=raster_in, out_path=None, get_ds=True, **kwargs)
        
    # 调用extract函数
    return extract(raster_in=raster_in, dst_in=ds, out_path=out_path, get_ds=get_ds)
    


def zonal_u(raster_in, dst_in, stats,dic=None,**kwargs):
    '''
    分区统计
    含临时统一操作
    分区栅格应为整型栅格
    
    Parameters
    ----------
    raster_in : TYPE
        输入栅格
    dst_in : TYPE
        分区数据栅格
    stats : 
       统计类型。基于df.agg(stats). e.g. 'mean' or ['mean','sum','max']...
    
    dic : dict
        分区数据栅格各值对应属性，默认为值本身
    **kwargs : TYPE
       unify可填入的其他参数

    Returns
    -------
    dataframe
        所需统计值的dataframe

    '''
    
    ds = unify(dst_in, raster_in, out_path=None, **kwargs)
    return zonal(raster_in=raster_in,dst_in=ds, stats=stats,dic=dic)
    
    
            





if __name__ == '__main__':
    
    
    raster_in = r'F:/PyCharm/pythonProject1/arcmap/015温度/土地利用/landuse_4y/1981-5km-tiff.tif'

    dst_in = r'F:\PyCharm\pythonProject1\arcmap\007那曲市\data\eva平均\eva_2.tif'

    out_path = r'F:\PyCharm\pythonProject1\代码\mycode\测试文件\1981-5km-tiff3.tif'
    
    out_path1 = r'F:\PyCharm\pythonProject1\arcmap\015温度\zonal\grand_average.xlsx'

    

    # ds = unify(out_path, raster_in, how='mode',Extract=True)

    ds = unify(raster_in, dst_in, how='mode',Extract=1)
    
    
    # dst = extract(raster_in=ds, dst_in=dst_in)
    out_ds(ds, out_path)
    
    
    
    check(ds,dst_in)
    # df = zonal(out_path,dst_in, ['mean','max'],dic={1:'森林'})

    # extract(raster_in, dst_in)

    # shape = get_RasterArrt(dst_in,'shape')

    # ds_pro = reproject(raster_in=raster_in, dst_in=dst_in)
    
    # ds_clip = clip(raster_in=ds_pro, dst_in=dst_in,Extract=True)
    
    # ds = resampling(raster_in=ds_clip,re_shape = shape)
    
    # df = zonal_u(raster_in, out_path,['mean','max'])
    # check(ds,dst_in)



    





















