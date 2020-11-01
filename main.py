# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 15:37:18 2020

@author: Du Bin
"""
import time
import numpy as np
import skimage
from skimage import io
import optargv
import optfuntion
import convfuntion

if __name__=='__main__':
    #read options from commmand line
    parser=optargv.optional_argv()
    (options,args)=parser.parse_args()
    
    #read image and convert to [0,1] with datatype=float
    image=skimage.io.imread(options.image_path)/256
    io.imshow(image)
    image=np.array(image,dtype=np.float32)
    row,col=image.shape
    u_len=row*col
    #true image vector
    u_true=image.flatten(order='F')
    u_true=u_true.T
    u_true=u_true.reshape(u_len,1)
    
    #add noise
    
    #try to split image in order to solve one by one,every part is 16*16
    part_size=16
    A=convfuntion.convulution_to_matrix(options.kernel_width, \
                                                options.gaussian_sigma, (part_size,part_size))
    W_x=convfuntion.partial_x(part_size, part_size)
    W_y=convfuntion.partial_y(part_size, part_size)
    W=W_x+W_y
    AT=A.T
    ATA=AT*A
    WT=W.T
    WTW=WT*W
    for i in range(col/part_size):
        for j in range(row/part_size):
            u_part_len=part_size**2
            u_part=image[i*part_size:(i+1)*part_size,j*part_size:(j+1)\
                         *part_size].flatten(order='F')
            f=A*u_part+np.random.normal(0,options.noise_level,(u_part_len,1))
            ATf=AT*f
            b0=np.zeros((u_part_len,1))
            d0=np.zeros_like(b0)
            A_left=
            for k in range(options.maxiter):
                
            
                
    
    #A=convfuntion.partial_x(row,col)  

