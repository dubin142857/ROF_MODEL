# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 19:51:47 2020

@author: Admin
"""
from optparse import OptionParser

def optional_argv():
    usage = "usage:%prog [options] arg1 arg2"
    parser = OptionParser(usage=usage)
    parser.add_option("-p","--image_path",action="store",
                  dest="image_path",default="image/peppers256.png",
                  help="input the path of images")
    parser.add_option("--lambda_weight",type="float",dest="lambda_weight",
                  default=2019,help="use to adjust the norm of gradient ")
    parser.add_option("--kernel_width",type="float",dest="kernel_width",
                  default=15,help="use to adjust the size of gauss kernel")
    parser.add_option("--gaussian_sigma",type=float,dest="gaussian_sigma",
                      default=1.0,help="use to adjust gauss kenel variance")
    parser.add_option("--output_image",dest="output_image",help="give the output image path")
    parser.add_option("--mu_weight",type="float",dest="mu_weight",default=2019,
                  help="use to adjust penalty term")
    parser.add_option("--delta",type="float",dest="delta",default=1.0,
                  help="use to adjust the stepsize of b")
    parser.add_option("--noise_level",type="float",dest="noise_level",default=0.01,
                  help="use to adjust the level of noise")
    parser.add_option("--maxiter",type="int",dest="maxiter",default=300,
                  help="use to adjust the maxinum number of iterions")
    parser.add_option("--device",dest="device",default="cpu",help="the information of device")
    parser.add_option("--tol",type="float",dest="tol",default=1e-4,help="the iteration error")
    parser.add_option("--sim_win_size",type="float",default=15,help="use to adjust the size of ssim")
    parser.add_option("--random_seed",type="float",default=-1,help="use to adjust random seed")
    return parser

