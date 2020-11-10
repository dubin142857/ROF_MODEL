# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 08:54:27 2020

@author: Admin
"""
import numpy as np
from skimage.measure import compare_ssim


    

def soft_threshold(tao,x):
    norm_x=np.linalg.norm(x,'fro')
    y=(x/norm_x*np.max([norm_x-tao,0])).copy()
    return y

def psnr(u_true,u):
    psnr_u=10*np.log10((u.max())**2/(((u-u_true)**2).mean()))
    return psnr_u

def ssim(u,u_true,win_size=15,data_range=1):
    return compare_ssim(u_true,u,win_size=win_size,data_range=data_range)

        
