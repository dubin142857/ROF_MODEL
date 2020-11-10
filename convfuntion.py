# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 19:42:54 2020

@author: Admin
"""
import numpy as np
def fspeical(row,col,gauss_sigma):
    """generate row*col gauss kernel"""
    k=np.zeros((row,col))
    middle=(row+1)/2
    sigma2=gauss_sigma**2
    for i in range(row):
        for j in range(col):
            k[i,j]=np.exp(-((i+1-middle)**2+(j+1-middle)**2)/(2*sigma2))
    k=k/np.sum(k)
    return k
def imfilter(image1,kernel):
    """compute gauss blur kernel*image1"""
    image=image1.copy()
    row=kernel.shape[0]#15
    mid=int(row/2)#7
    #填补周期边界
    observe_image=np.zeros((image.shape[0]+row-1,image.shape[1]+row-1))
    observe_image[0:mid,0:mid]=image[-mid:,-mid:].copy()
    observe_image[0:mid,mid:-mid]=image[-mid:,:].copy()
    observe_image[0:mid,-mid:]=image[-mid:,0:mid].copy()
    observe_image[mid:-mid,0:mid]=image[:,-mid:].copy()
    observe_image[mid:-mid,mid:-mid]=image.copy()
    observe_image[mid:-mid,-mid:]=image[:,0:mid].copy()
    observe_image[-mid:,0:mid]=image[0:mid,-mid:].copy()  
    observe_image[-mid:,mid:-mid]=image[0:mid,:].copy()
    observe_image[-mid:,-mid:]=image[0:mid,0:mid].copy()
    #进行卷积
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i,j]=np.sum(observe_image[i:i+row,j:j+row]*kernel)
    return image
def zero_pad(image,shape,position='corner'):
    """extend image to a certain size with zeros"""
    shape=np.asarray(shape,dtype=int)
    imshape=np.asarray(image.shape,dtype=int)
    if np.alltrue(imshape==shape):
        return image
    if np.any(shape<=0):
        raise ValueError("ZERO_PAD:null or negative shape given")
    
    dshape=shape-imshape
    if np.any(dshape<0):
        raise ValueError("ZERO_PAD:target size smaller than source one")
    
    pad_img=np.zeros(shape,dtype=image.dtype)
    idx,idy=np.indices(imshape)
    
    if position=='center':
        if np.any(dshape%2!=0):
            raise ValueError("ZERO_PAD:source and target shapes have "
                             "different parity.")
        offx,offy = dshape//2
    else:
        offx,offy=(0,0)
    
    pad_img[idx+offx,idy+offy]=image.copy()
    return pad_img
def psf2otf(psf,outSize):
    """refer to MATlab code: function psf2otf"""
    if np.all(psf==0):
        return np.zeros_like(psf)
    psfSize = np.array(psf.shape)
    outSize = np.array(outSize)
    padSize = outSize - psfSize
    psf = np.pad(psf, ((0, padSize[0]), (0, padSize[1])), 'constant')
    for i in range(len(psfSize)):
        psf = np.roll(psf, -int(psfSize[i] / 2), i)
    otf = np.fft.fft2(psf)
    nElem = np.prod(psfSize)
    nOps = 0
    for k in range(len(psfSize)):
        nffts = nElem / psfSize[k]
        nOps = nOps + psfSize[k] * np.log2(psfSize[k]) * nffts
    if np.max(np.abs(np.imag(otf))) / np.max(np.abs(otf)) <= nOps * np.finfo(np.float32).eps:
        otf = np.real(otf)
    return otf
def dive(X,Y):
    """transpose of the forward finite difference operator"""
    fwd_diff_rowX=np.expand_dims(X[:,-1]-X[:,0],axis=1)
    DtXY=np.concatenate((fwd_diff_rowX,-np.diff(X,1,1)),axis=1)
    fwd_diff_colX=np.expand_dims(Y[-1,:]-Y[0,:],axis=0)
    DtXY=DtXY+np.concatenate((fwd_diff_colX,-np.diff(Y,1,0)),axis=0)
    return DtXY

def getC(image,kernel):
    sizeF=np.shape(image)
    #卷积的傅里叶变换
    eigsK=psf2otf(kernel,sizeF)
    #卷积的转置乘上图像
    KtF=np.real(np.fft.ifft2(np.conj(eigsK)*np.fft.fft2(image)))
    diff_kernelX=np.expand_dims(np.array([1,-1]),axis=0)
    diff_kernelY = np.array([[1],[-1]])
    eigsDtD=np.abs(psf2otf(diff_kernelX, sizeF))**2+np.abs(psf2otf(diff_kernelY,sizeF))**2
    eigsKtK=np.abs(eigsK)**2
    return (eigsK,KtF,eigsDtD,eigsKtK)

def forward(u):
    #向前差分算子
    end_col_diff=np.expand_dims(u[:,0]-u[:,-1],axis=1)
    end_row_diff=np.expand_dims(u[0,:]-u[-1,:],axis=0)
    Dux=np.concatenate((np.diff(u,1,1),end_col_diff),axis=1)
    Duy=np.concatenate((np.diff(u,1,0),end_row_diff),axis=0)
    return (Dux,Duy)
def fval(D1X,D2X,eigsK,X,img,lambda_weight):
    """计算ROF最优化的函数值"""
    f=np.sum(np.abs(D1X)+np.abs(D2X))*lambda_weight
    KXF=np.real(np.fft.ifft2(eigsK*np.fft.fft2(X)))-img
    f=f+0.5*np.linalg.norm(KXF,'fro')**2
    return f
    
    
    





           