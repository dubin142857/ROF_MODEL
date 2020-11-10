# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 19:51:47 2020

@author: Admin
"""
import sys, getopt

def  default_options():
    options = {
        "--image_path":"image/peppers256.png",
        "--lambda_weight":1,
        "--kernel_width":15,
        "--gaussian_sigma":1,
        "--output_image":"None",
        "--mu_weight":2**5,
        "--delta":1,
        "--noise_level":0.01,
        "--maxiter":300,
        "--device":'cpu',
        "--tol":1*1e-4,
        "--ssim_win_size":15,
        "--iprint":1
        }
    return options

def _setopt(options):
    options.pop('-f',1)
    default =default_options()
    for k in default:
        if isinstance(default[k],(float,int)): 
            if isinstance(options[k],str):
                options[k]=eval(options[k])
            if isinstance(default[k], float):
                options[k]=float(options[k])
            else:
                options[k]=int(options[k])
        else:
            options[k]=type(default[k])(options[k])
    return None
def setoptions(*,argv=None,kw=None):
    options=default_options()
    longopts=list(k[2:]+'=' for k in options)
    argv =({} if argv is None else
           dict(getopt.getopt(argv,shortopts='f',longopts=longopts)[0]))
    kw=({} if kw is None else kw)
    
    options.update(kw)
    options.update(argv)
    
    _setopt(options)
    
    globalnames={}
    for k,v in options.items():
        globalnames[k[2:]]=v
    return options,globalnames