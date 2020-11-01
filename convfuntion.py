# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 19:42:54 2020

@author: Admin
"""


import numpy as np

def matrix_fill_diagonal(value,diag_n,A):
    """use to fill the diag_n sub_diagonal of row*col matrix  A,return A
    default diagonal is 0,up is positive"""
    row,col=A.shape
    if diag_n>0:
        for i in range(col-diag_n):
            A[i,i+diag_n]=value
    else:
        for i in range(row+diag_n):
            A[i+diag_n,i]=value
    return A

def dim2_to_dim1(dim2_coordinate,m,n):
    """use to transform m*n matrix A to vector u,which is arranged by columns."""
    x,y=dim2_coordinate
    x=x%m
    y=y%n
    return x+y*n
def dim1_to_dim2(dim1_coordinate,m,n):
    """use to transform m*n vector u to m*n matrix A,return corresponding 
    matrix corrdinate"""  
    x=dim1_coordinate%n
    y=int(dim1_coordinate/n)
    return [x,y]
def convulution_to_matrix(kernel_width,gaussian_sigma,image_size):
    """
    this code use to transform gauss_kernel with variance 'gaussian_sigma' to
    a s*s matrix. Above s =m*n,which (m,n)is the size of image.return matrix A.
    arguments:
        kernel_width: [float] gauss_kernel width
        gauss_sigma:  [float] gauss_kernel variance
        image_size:  [list]  the size of the image that blurred by gauss_kernel
    """
    m,n=image_size
    rows=m*n
    gauss_kernel=np.zeros((kernel_width,kernel_width)).astype(np.float32)
    gauss_means=(kernel_width+1)/2
    for i in range(kernel_width):
        for j in range(kernel_width):
            gauss_kernel[i,j]=np.exp(-((i+1-gauss_means)**2+(j+1-gauss_means)**2)\
                                     /(gaussian_sigma**2)).astype(np.float32)
    gauss_kernel=gauss_kernel/np.sum(gauss_kernel)
    A=np.zeros((rows,rows)).astype(np.float32)
    middle=int(kernel_width/2)
    for i in range(rows):
        x,y=dim1_to_dim2(i,m,n)
        for j in range(kernel_width):
            for k in range(kernel_width): 
                A[i,dim2_to_dim1((x+j-middle,y+k-middle),m,n)]=gauss_kernel(j,k)
    return A

def convulution_transpose_to_matrix(kernel_width,gaussian_sigma,image_size):
    """
    this code use to transform gauss_kernel's transpose with variance 'gaussian_sigma' to
    a s*s matrix. Above s =m*n,which (m,n)is the size of image.return matrix A.
    arguments:
        kernel_width: [float] gauss_kernel width
        gauss_sigma:  [float] gauss_kernel variance
        image_size:  [list]  the size of the image that blurred by gauss_kernel
    """
    m,n=image_size
    rows=m*n
    gauss_kernel=np.zeros((kernel_width,kernel_width)).astype(np.float32)
    gauss_means=(kernel_width+1)/2
    for i in range(kernel_width):
        for j in range(kernel_width):
            gauss_kernel[i,j]=np.exp(-((i+1+gauss_means)**2+(j+1+gauss_means)**2)\
                                     /(gaussian_sigma**2)).astype(np.float32)
    gauss_kernel=gauss_kernel/np.sum(gauss_kernel)
    A=np.zeros((rows,rows)).astype(np.float32)
    middle=int(kernel_width/2)
    for i in range(rows):
        x,y=dim1_to_dim2(i,m,n)
        for j in range(kernel_width):
            for k in range(kernel_width): 
                A[i,dim2_to_dim1((x+j-middle,y+k-middle),m,n)]=gauss_kernel(j,k)
    return A

def partial_x(m,n):
    """use to generate m*n matrix A,which is the matrix form of partial x
    partial_x = x_{i+1}-x_{i}"""
    A=np.zeros((m,n)).astype(np.int8)
    A=matrix_fill_diagonal(-1,0,A)
    A=matrix_fill_diagonal(1,m*(n-1),A)
    A=matrix_fill_diagonal(1,m,A)
    return A
    
def partial_y(m,n):
    """use to generate m*n matrix A,which is the matrix form of partial y
    partial_y=y_{i+1}-y_{i}"""                
    A=np.zeros((m,n)).astype(np.int8)
    A=matrix_fill_diagonal(-1,0,A)
    A=matrix_fill_diagonal(1,1,A)
    for i in range(n):
        k=i*m
        A[k+m-1,k]=1
    return A

                    
            
    