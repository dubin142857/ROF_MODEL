# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 15:37:18 2020

@author: Du Bin
"""
import sys
import time
import numpy as np
import skimage
from skimage import io
import optargv
import optfuntion
import convfuntion
import warnings

warnings.filterwarnings("ignore")
# read data from command line
if len(sys.argv)>1 and (sys.argv[1]=="--help"or sys.argv[1]=="-h"):
    options=optargv.default_options()
    print("Usage:")
    print("      python main.py opt1=value1 opt2=value2...")
    print("      e.g python main.py --image_path=image/peppers256.png\n") 
    print("Available options:default values")
    for k,v in options.items():
        print("",k,":",v)
    sys.exit()
options,globalnames = optargv.setoptions(argv=sys.argv[1:],kw=None)
globals().update(globalnames)

#read image and convert to [0,1] 
f=skimage.io.imread(image_path)/256
f=np.asarray(f,dtype=float)
u_true =f.copy()
f_row=int(f.shape[0])
f_col=int(f.shape[1])
gauss_kernel=convfuntion.fspeical(kernel_width,kernel_width,gaussian_sigma)
f=convfuntion.imfilter(f,gauss_kernel)+np.random.randn(f_row,f_col)*noise_level
norm_f=np.linalg.norm(f,'fro')
#给定d0,b0初始值
d=np.zeros(f.shape)
D1f,D2f=convfuntion.forward(f)
d=(D1f+D2f).copy()
b=np.zeros_like(d)
eigsK,KtF,eigsDtD,eigsKtK =convfuntion.getC(f,gauss_kernel)
abs_error=[]
rel_error=[]
psnr=[]
ssim=[]
for k in range(maxiter):
    
    u=np.fft.fft2(KtF+mu_weight*convfuntion.dive(d-b))/(eigsKtK+mu_weight*eigsDtD)
    u=np.real(np.fft.ifft2(u))
    #更新d,b
    diff_u1,diff_u2=convfuntion.forward(u)
    diff_u=(diff_u1+diff_u2).copy()
    tao=lambda_weight/mu_weight
    d=optfuntion.soft_threshold(tao,diff_u+b)
    wu_d=(diff_u-d).copy()
    b=b+delta*(wu_d)
    abs_error.append(np.linalg.norm(wu_d,'fro'))
    rel_error.append(abs_error[-1]/norm_f)
    psnr.append(optfuntion.psnr(u_true,u))
    ssim.append(optfuntion.compare_ssim(u_true,u,ssim_win_size,1))
    if iprint>0 and k%iprint==0:
            print("iteration: {0}".format(k))
            print("     ||Wu-d||: {:.5e}".format(abs_error[-1]))
            print("         PSNR: {:.3f}".format(psnr[-1]))
    if rel_error[-1]<tol and abs_error[-1]<1e-2:
        break
psnr.append(optfuntion.psnr(u_true,f))
ssim.append(optfuntion.compare_ssim(u_true,f,ssim_win_size,1))    
if  output_image == "None":
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1,3,1)
    ax.imshow(u_true, cmap='gray')
    ax.set_title("u_true", fontsize=20); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(1,3,2)
    ax.imshow(f, cmap='gray')
    ax.set_title("f", fontsize=20); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(1,3,3)
    ax.imshow(u, cmap='gray')
    ax.set_title("u", fontsize=20); ax.set_xticks([]); ax.set_yticks([])
    fig.set_size_inches([15,5])
    fig.savefig(output_image, dpi=200)
    # ax.plot()
    
    


    
    
        