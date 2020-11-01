# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 08:54:27 2020

@author: Admin
"""
import numpy as np
from skimage.measure import compare_ssim

def conjugate_gradient(A,b,x0=None,tol=1e-5,maxiter=20):
    """
    use conjugate_gadient method to solve equation Ax=b,return x.
    agruments:
        x0:initial value;
        tol:iteration error;
        maxiter: maxinum number of iteration.
    """
    A_cols = A.shape[1]
    if x0 is None:
        x0=np.zeros((A_cols,1))
    x=x0
    k=0
    r=b-np.dot(A,x)
    rou=np.dot(r,r)
    rou1=rou
    while np.sqrt(rou)>tol*np.linalg.norm(b) and k<maxiter:
        k=k+1
        if k==1:
            p=r
        else:
            belta=rou/rou1
            p=r+belta*p
        w=np.dot(A,p)
        alpha=rou/np.dot(p,w)
        x=x+alpha*p
        r=r-alpha*w
        rou1=rou
        rou=np.dot(r,r)
    return x,k

def soft_threshold(tao,x):
    norm_x=np.linalg.norm(x)
    y=x/norm_x*np.max([norm_x-tao,0])
    return y

def psnr(u,u_true):
    psnr_u=10*np.log10((u.max)**2/(((u-u_true)**2).mean()))
    return psnr_u

def ssim(u,u_true,win_size=15,data_range=1):
    return compare_ssim(u,u_true,win_size=win_size,data_range=data_range)

        
