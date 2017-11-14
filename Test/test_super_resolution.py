
#giang change
from __future__ import print_function
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.objectives import squared_error
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc as scimc
import PIL
import math
import scipy.misc
import scipy.io as sio
import string
import h5py
import time
import cPickle as pickle

from hashlib import sha1

from numpy import all, array, uint8

from lasagne.layers import Conv2DLayer, ElemwiseSumLayer, batch_norm, InputLayer,MaxPool2DLayer,TransposedConv2DLayer,prelu
from lasagne.nonlinearities import rectify, identity
from lasagne.init import GlorotNormal, Constant

from PIL import Image

from numpy.linalg import inv

import cPickle as pickle

import sys
from os import listdir
from os.path import isfile, join
import os

# Track 1 Testing
# SCALE = 3
# PRE_TRAINED_WEIGHTS = './pre-trained/Track1/models_s_x3/cvprdata_11700.params'
# INPUT_DIR = '/media/titanx2/6B3C546A318EAD3E/DI2K/Testing/Track1/DIV2K_test_LR_bicubic/X3'
# OUPUT_DIR = '/media/titanx2/6B3C546A318EAD3E/DI2K/Testing/Track1/DIV2K_test_LR_bicubic/X3_res/'

# SCALE = 4
# PRE_TRAINED_WEIGHTS = './pre-trained/Track1/models_s_x4/cvprdata_11700.params'
# INPUT_DIR = '/media/titanx2/6B3C546A318EAD3E/DI2K/Testing/Track1/DIV2K_test_LR_bicubic/X4'
# OUPUT_DIR = '/media/titanx2/6B3C546A318EAD3E/DI2K/Testing/Track1/DIV2K_test_LR_bicubic/X4_res/'


# Track 2 Testing
# SCALE = 2
# PRE_TRAINED_WEIGHTS = './pre-trained/Track2/models_s_x2/track2_scale_2_pretrained_weights.params'
# INPUT_DIR = '/media/titanx2/6B3C546A318EAD3E/DI2K/Testing/Track2/DIV2K_test_LR_unknown/X2'
# OUPUT_DIR = '/media/titanx2/6B3C546A318EAD3E/DI2K/Testing/Track2/DIV2K_test_LR_unknown/X2_res/'

# SCALE = 3
# PRE_TRAINED_WEIGHTS = './pre-trained/Track2/models_s_x3/track2_scale_3_pretrained_weights.params'
# INPUT_DIR = '/media/titanx2/6B3C546A318EAD3E/DI2K/Testing/Track2/DIV2K_test_LR_unknown/X3'
# OUPUT_DIR = '/media/titanx2/6B3C546A318EAD3E/DI2K/Testing/Track2/DIV2K_test_LR_unknown/X3_res/'

SCALE = 4
PRE_TRAINED_WEIGHTS = './pre-trained/Track2/models_s_x4/track2_scale_4_pretrained_weights.params'
INPUT_DIR = '/media/titanx2/6B3C546A318EAD3E/DI2K/Testing/Track2/DIV2K_test_LR_unknown/X4'
OUPUT_DIR = '/media/titanx2/6B3C546A318EAD3E/DI2K/Testing/Track2/DIV2K_test_LR_unknown/X4_res/'



def read_model_data(net1, filename):
    """
        Load pre-trained weights from files
        Args:
            - net1: Lasagne network
            - filename(string): pre-trained weight
    """
    
    c=0

    with open (filename, 'rb') as fp:
        data = pickle.load(fp)

        for l in xrange(1,5):
            for i in xrange(1,21):            
                lname = 'conv{}_{}'.format(i,l)
                lprelu = 'prelu{}_{}'.format(i,l)
                net1[lname].W.set_value(data[c])
                c=c+1
                net1[lname].b.set_value(data[c])
                c=c+1
                if i < 20:
                    net1[lprelu].alpha.set_value(data[c])
                    c=c+1

        net1['scale_1'].scales.set_value(data[c])
        c=c+1
        net1['scale_2'].scales.set_value(data[c])
        c=c+1
        net1['scale_3'].scales.set_value(data[c])
        c=c+1
        net1['scale_4'].scales.set_value(data[c])

def try_an_image_validation_mode(SCALE,data_file,gt_file,result_dir):
    """
        Test validation image
        Args:
            - SCALE(int): scale factor
            - data_file(string): input low-resolution image
            - gt_file(string): groundtrust high-resolution image
            - result_dir(string): a directory for saving the result
    """

    img = scimc.imread(data_file,  mode = 'RGB')    
    gt = scimc.imread(gt_file,mode = 'RGB')    
    
    H,W,_ = img.shape 
    

    input_img = np.zeros(gt.shape,dtype=np.float32)

    try:
        input_img[:,:,0] =  scimc.imresize(img[:,:,0],(SCALE*img.shape[0],SCALE*img.shape[1]),interp='bicubic', mode = 'F')
        input_img[:,:,1] =  scimc.imresize(img[:,:,1],(SCALE*img.shape[0],SCALE*img.shape[1]),interp='bicubic', mode = 'F')
        input_img[:,:,2] =  scimc.imresize(img[:,:,2],(SCALE*img.shape[0],SCALE*img.shape[1]),interp='bicubic', mode = 'F')
                    
        data = input_img/255.0
        data = np.transpose(data,(2,0,1))
        

        super_res = probs(data[np.newaxis,:,:,:])
        super_res = np.squeeze(super_res)
        super_res=np.clip(super_res,0,1)        
        super_res = np.round(255*super_res)
        super_res = np.transpose(super_res,(1,2,0))            

        tmp = super_res -gt
        tmp = tmp[6+SCALE:-(6+SCALE), 6+SCALE:-(6+SCALE),:]
        
        mse = np.sum(tmp**2)/(tmp.shape[0]*tmp.shape[1]*tmp.shape[2])

        psnr = 20*math.log10(255.0) - 10*math.log10(mse)
        print(data_file,':',psnr)
        
        
    except ValueError:
        print("Error reading file: ",data_file)
        exit(1)


    
    seq = data_file.rsplit('/')

    super_res = super_res.astype(np.uint8)
    scimc.imsave(result_dir+seq[len(seq)-1],super_res)
    
    
    return psnr

def try_an_image_testing_mode(SCALE,data_file,result_dir):
    """
        Test an image
        Args:
            - SCALE(int): scale factor
            - data_file(string): input low-resolution image            
            - result_dir(string): a directory for saving the result
    """

    img = scimc.imread(data_file,  mode = 'RGB')    
    
    
    H,W,_ = img.shape 

    input_img = np.zeros((SCALE*img.shape[0],SCALE*img.shape[1],img.shape[2]),dtype=np.float32)       

    try:
        input_img[:,:,0] =  scimc.imresize(img[:,:,0],(SCALE*img.shape[0],SCALE*img.shape[1]),interp='bicubic', mode = 'F')
        input_img[:,:,1] =  scimc.imresize(img[:,:,1],(SCALE*img.shape[0],SCALE*img.shape[1]),interp='bicubic', mode = 'F')
        input_img[:,:,2] =  scimc.imresize(img[:,:,2],(SCALE*img.shape[0],SCALE*img.shape[1]),interp='bicubic', mode = 'F')
                    
        data = input_img/255.0
        data = np.transpose(data,(2,0,1))
        

        super_res = probs(data[np.newaxis,:,:,:])
        super_res = np.squeeze(super_res)
        super_res=np.clip(super_res,0,1)        
        super_res = np.round(255*super_res)
        super_res = np.transpose(super_res,(1,2,0))                    
        
    except ValueError:
        print("Error reading file: ",data_file)
        exit(1)

    seq = data_file.rsplit('/')

    super_res = super_res.astype(np.uint8)
    scimc.imsave(result_dir+seq[len(seq)-1],super_res)
    

def batch_test_validation_mode(SCALE,lowImgPath,highImgPath,saveResPath):
    """
        Test a set of images
        Args:
            - SCALE(int): scale factor
            - lowImgPath(string): a folder of input low-resolution images
            - highImgPath(string): a folder of high-resolution images
            - saveResPath(string): a directory for saving the result
    """
    onlyfiles = [f for f in listdir(lowImgPath) if isfile(join(lowImgPath, f))]
    avg_psnr = 0
    for f in onlyfiles:  
        gt = "{}.png".format(f[0:4])        
        psnr = try_an_image_validation_mode(SCALE,join(lowImgPath, f), join(highImgPath, gt), saveResPath)        
        avg_psnr = avg_psnr + psnr        
    avg_psnr = avg_psnr/len(onlyfiles)
    print('OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOH. Average psnr of {}: {}'.format(lowImgPath,avg_psnr))

def batch_test_testing_mode(SCALE,lowImgPath,saveResPath):
    """
        Test a set of images.
        Args:
            - SCALE(int): scale factor
            - lowImgPath(string): a folder of input low-resolution images
            - saveResPath(string): a directory for saving the result
    """
    onlyfiles = [f for f in listdir(lowImgPath) if isfile(join(lowImgPath, f))]
    global_starting_time = time.time()
    for f in onlyfiles:            
        try_an_image_testing_mode(SCALE,join(lowImgPath, f), saveResPath)        
    global_ending_time = time.time()

    print('Average running time: {}'.format((global_ending_time - global_starting_time)*1.0/len(onlyfiles)))



input_values = T.tensor4('X')

# construct CNN net    
net1 = {}

net1['input']  = lasagne.layers.InputLayer((None, 3, None, None),input_values,name = 'input')  
############# VDSR 1 ##################################

net1['conv1_1']  = lasagne.layers.Conv2DLayer(net1['input'],64,(3,3),pad = 1, name = 'conv1_1', nonlinearity =  lasagne.nonlinearities.rectify, W=lasagne.init.GlorotNormal(gain=math.sqrt(2)),b=lasagne.init.Constant(0.))        
net1['prelu1_1'] = prelu(net1['conv1_1'],name='prelu1_1')    
       
for i in xrange(1,19): 
    namelayer ='conv{}_1'.format(i+1)
    prvlayername = 'prelu{}_1'.format(i)
    net1[namelayer] =  lasagne.layers.Conv2DLayer(net1[prvlayername],64,(3,3),pad = 1, name = namelayer,nonlinearity =  lasagne.nonlinearities.rectify,W=lasagne.init.GlorotNormal(gain=math.sqrt(2)),b=lasagne.init.Constant(0.))
    net1['prelu{}_1'.format(i+1)] = prelu(net1[namelayer],name='prelu{}_1'.format(i+1))

net1['conv20_1'] = lasagne.layers.Conv2DLayer(net1['prelu19_1'],3,(3,3),pad = 1, name = 'conv20_1',nonlinearity = lasagne.nonlinearities.identity, W=lasagne.init.GlorotNormal(gain=math.sqrt(2)),b=lasagne.init.Constant(0.))        
net1['sum_1']= lasagne.layers.ElemwiseSumLayer({net1['conv20_1'], net1['input']}, name ='sum_1')

############# VDSR 2 ##################################
net1['conv1_2']  = lasagne.layers.Conv2DLayer(net1['sum_1'],64,(3,3),pad = 1, name = 'conv1_2', nonlinearity =  lasagne.nonlinearities.rectify, W=lasagne.init.GlorotNormal(gain=math.sqrt(2)),b=lasagne.init.Constant(0.))        
net1['prelu1_2'] = prelu(net1['conv1_2'],name='prelu1_2')    
       
for i in xrange(1,19): 
    namelayer ='conv{}_2'.format(i+1)
    prvlayername = 'prelu{}_2'.format(i)
    net1[namelayer] =  lasagne.layers.Conv2DLayer(net1[prvlayername],64,(3,3),pad = 1, name = namelayer,nonlinearity =  lasagne.nonlinearities.rectify,W=lasagne.init.GlorotNormal(gain=math.sqrt(2)),b=lasagne.init.Constant(0.))
    net1['prelu{}_2'.format(i+1)] = prelu(net1[namelayer],name='prelu{}_2'.format(i+1))

net1['conv20_2'] = lasagne.layers.Conv2DLayer(net1['prelu19_2'],3,(3,3),pad = 1, name = 'conv20_2',nonlinearity = lasagne.nonlinearities.identity, W=lasagne.init.GlorotNormal(gain=math.sqrt(2)),b=lasagne.init.Constant(0.))           
net1['sum_2']= lasagne.layers.ElemwiseSumLayer({net1['conv20_2'], net1['sum_1']}, name ='sum_2')

############# VDSR 3 ##################################
net1['conv1_3']  = lasagne.layers.Conv2DLayer(net1['sum_2'],64,(3,3),pad = 1, name = 'conv1_3', nonlinearity =  lasagne.nonlinearities.rectify, W=lasagne.init.GlorotNormal(gain=math.sqrt(2)),b=lasagne.init.Constant(0.))           
net1['prelu1_3'] = prelu(net1['conv1_3'],name='prelu1_3')    
       
for i in xrange(1,19): 
    namelayer ='conv{}_3'.format(i+1)
    prvlayername = 'prelu{}_3'.format(i)
    net1[namelayer] =  lasagne.layers.Conv2DLayer(net1[prvlayername],64,(3,3),pad = 1, name = namelayer,nonlinearity =  lasagne.nonlinearities.rectify,W=lasagne.init.GlorotNormal(gain=math.sqrt(2)),b=lasagne.init.Constant(0.))
    net1['prelu{}_3'.format(i+1)] = prelu(net1[namelayer],name='prelu{}_3'.format(i+1))

net1['conv20_3'] = lasagne.layers.Conv2DLayer(net1['prelu19_3'],3,(3,3),pad = 1, name = 'conv20_3',nonlinearity = lasagne.nonlinearities.identity, W=lasagne.init.GlorotNormal(gain=math.sqrt(2)),b=lasagne.init.Constant(0.))    
   
net1['sum_3']= lasagne.layers.ElemwiseSumLayer({net1['conv20_3'], net1['sum_2']}, name ='sum_3')

############# VDSR 4 ##################################
net1['conv1_4']  = lasagne.layers.Conv2DLayer(net1['sum_3'],64,(3,3),pad = 1, name = 'conv1_4', nonlinearity =  lasagne.nonlinearities.rectify, W=lasagne.init.GlorotNormal(gain=math.sqrt(2)),b=lasagne.init.Constant(0.))        
net1['prelu1_4'] = prelu(net1['conv1_4'],name='prelu1_4')    
       
for i in xrange(1,19): 
    namelayer ='conv{}_4'.format(i+1)
    prvlayername = 'prelu{}_4'.format(i)
    net1[namelayer] =  lasagne.layers.Conv2DLayer(net1[prvlayername],64,(3,3),pad = 1, name = namelayer,nonlinearity =  lasagne.nonlinearities.rectify,W=lasagne.init.GlorotNormal(gain=math.sqrt(2)),b=lasagne.init.Constant(0.))
    net1['prelu{}_4'.format(i+1)] = prelu(net1[namelayer],name='prelu{}_4'.format(i+1))

net1['conv20_4'] = lasagne.layers.Conv2DLayer(net1['prelu19_4'],3,(3,3),pad = 1, name = 'conv20_4',nonlinearity = lasagne.nonlinearities.identity, W=lasagne.init.GlorotNormal(gain=math.sqrt(2)),b=lasagne.init.Constant(0.))    
   
net1['sum_4']= lasagne.layers.ElemwiseSumLayer({net1['conv20_4'], net1['sum_3']}, name ='sum_4')

############# fusion  ##################################

net1['scale_1'] = lasagne.layers.ScaleLayer(net1['sum_1'],scales=lasagne.init.Constant(0.2),name='scale_1')
net1['scale_2'] = lasagne.layers.ScaleLayer(net1['sum_2'],scales=lasagne.init.Constant(0.2),name='scale_2')
net1['scale_3'] = lasagne.layers.ScaleLayer(net1['sum_3'],scales=lasagne.init.Constant(0.2),name='scale_3')
net1['scale_4'] = lasagne.layers.ScaleLayer(net1['sum_4'],scales=lasagne.init.Constant(0.2),name='scale_4')

net1['output']= lasagne.layers.ElemwiseSumLayer({net1['scale_1'], net1['scale_2'],net1['scale_3'],net1['scale_4']}, name ='output')

    
network_output = lasagne.layers.get_output(net1['output'])   
probs = theano.function([input_values], network_output)

read_model_data(net1,PRE_TRAINED_WEIGHTS)
       
print("Testing ...")

#batchTest(SCALE,'/media/titanx2/6B3C546A318EAD3E/DI2K/DIV2K_valid_LR_bicubic/X3','/media/titanx2/6B3C546A318EAD3E/DI2K/DIV2K_valid_LR_bicubic/DIV2K_valid_HR','/media/titanx2/6B3C546A318EAD3E/DI2K/DIV2K_valid_LR_bicubic/X3_res/')

batch_test_testing_mode(SCALE,INPUT_DIR,OUPUT_DIR)





