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

sys.path.insert(0,'/home/titanx2/Downloads/caffe-master/python')
import caffe

#net_caffe = caffe.Net('VDSR_net.prototxt', '_iter_VDSR_Official.caffemodel', caffe.TEST)
#layers_caffe = dict(zip(list(net_caffe._layer_names), net_caffe.layers))

MAX_PATCH_NUM = 300000
TEST_MODE = False

SCALE_BASE = 1.1

# Optimization learning rate
LEARNING_RATE = 1.0e-5

# All gradients above this will be clipped
GRAD_CLIP = 0.1

# Number of epochs to train the net
NUM_EPOCHS = 100

# Batch size
BATCH_SIZE = 48

STRIDE = 10

# input and output size
SIZE_INPUT = 48
SIZE_LABEL = 48


def write_model_data(net1, filename):
    """
        Write network weights to a file
        Args:
            - net1: Lasagne network
            - filename(string): pre-trained weight
    """
    data = []

    for l in xrange(1,5):
        for i in xrange(1,21):            
            lname = 'conv{}_{}'.format(i,l)
            lprelu = 'prelu{}_{}'.format(i,l)
            data.append(net1[lname].W.get_value())
            data.append(net1[lname].b.get_value())
            if i < 20:
                data.append(net1[lprelu].alpha.get_value())

    data.append(net1['scale_1'].scales.get_value())
    data.append(net1['scale_2'].scales.get_value())
    data.append(net1['scale_3'].scales.get_value())
    data.append(net1['scale_4'].scales.get_value())
    with open(filename, 'wb') as fp:
        pickle.dump(data, fp)

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


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

    

def recursive_load_files(mypath, files):
    """
        Get all files in a folder and its sub-folders
        Args:
            - mypath: Lasagne network
            - files(list of string): list of files
    """
    [files.append(join(mypath, f)) for f in listdir(mypath) if isfile(join(mypath, f))]
    for d in listdir(mypath):
        if os.path.isdir(join(mypath,d)):
            recursive_load_files(join(mypath,d), files)

def load_CVPR_dataset(SCALE,mypath,gtpath,stride=STRIDE):  
    """
        Get all files in a folder and its sub-folders
        Args:
            - SCALE: scale factor
            - mypath: a folder containing low res imges
            - gtpath: a folder containing high res imges            
            - stride(int): stride to extract patches in images
        output:
            - data(hdf5), label(hdf5)
    """
    files=[]
    recursive_load_files(mypath, files)

    N = MAX_PATCH_NUM
    label = np.zeros((N,3,SIZE_LABEL,SIZE_LABEL),dtype=np.float32)
    
    data = np.zeros((N,3,SIZE_INPUT,SIZE_INPUT),dtype=np.float32)

    print(SIZE_LABEL)

    n = 0
    for i,f in enumerate(files):

                
        img = scimc.imread(f, mode = 'RGB')        

        s = f.split('/')

        #print(s)

        label_img = scimc.imread(join(gtpath,s[len(s)-1][0:4]+s[len(s)-1][6:]), mode = 'RGB')            
        label_img = label_img.astype(np.float32)


        input_img = 0*label_img
        
        H,W,_ = input_img.shape 

        input_img[:,:,0] =  scimc.imresize(img[:,:,0],(SCALE*img.shape[0],SCALE*img.shape[1]),interp='bicubic', mode = 'F')
        input_img[:,:,1] =  scimc.imresize(img[:,:,1],(SCALE*img.shape[0],SCALE*img.shape[1]),interp='bicubic', mode = 'F')
        input_img[:,:,2] =  scimc.imresize(img[:,:,2],(SCALE*img.shape[0],SCALE*img.shape[1]),interp='bicubic', mode = 'F')
       
              
        for r in xrange(0,H-SIZE_INPUT-1,stride):
            for c in xrange(0,W-SIZE_INPUT-1,stride):
                
                if(n>=MAX_PATCH_NUM):
                    continue
                              
                #lap[n,0,:,:] = input_img[r : SIZE_INPUT + r, c : SIZE_INPUT + c]
                data[n,:,:,:] = np.transpose(input_img[r : SIZE_INPUT + r, c : SIZE_INPUT + c,:],(2,0,1))/255.0
                label[n,:,:,:] = np.transpose(label_img[r : SIZE_INPUT + r, c : SIZE_INPUT + c,:],(2,0,1))/255.0                                           
                n = n + 1   
                if n%1000==0:
                    print(n)

    print("n:",n)

    data = data[0:n,:,:,:]
    label = label[0:n,:,:,:]

    with h5py.File('/media/titanx2/6B3C546A318EAD3E/DI2K/DIV2K_train_LR_unknown/data_x3_64.h5', 'w') as hf:
        hf.create_dataset("data",  data=data)
    with h5py.File('/media/titanx2/6B3C546A318EAD3E/DI2K/DIV2K_train_LR_unknown/label_x3_64.h5', 'w') as hf:
        hf.create_dataset("label",  data=label)

    print("finishing")
   
    return data,label       

def load_hdf5_data(fname,dname):    
    with h5py.File(fname,'r') as hf:
        print('List of arrays in this file: \n', hf.keys())
        data = np.array(hf.get(dname))
            
    return data

def main(num_epochs = NUM_EPOCHS):

    vgg_offset = np.zeros((3,1,1),dtype=np.float32)
    vgg_offset[0,0,0] = 104.00698793
    vgg_offset[1,0,0] = 116.66876762
    vgg_offset[2,0,0] = 122.67891434

    

    input_values = T.tensor4('X')
    input_rep_vals = input_values/255.0

    # construct CNN net    
    w_lr = 1
    b_lr = 0.1
    net = {}
    net={}
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

        
    l_out = net1['output']
    # Theano tensor for the targets
    target_values = T.tensor4('target_output')
    
    network_output = lasagne.layers.get_output(l_out)    
    loss = 0.5*SIZE_LABEL**2*lasagne.objectives.squared_error(network_output, target_values).mean()
    print("Computing updates ...")

    params = lasagne.layers.get_all_params(l_out)
    grads = theano.grad(loss, params)
    for idx, param in enumerate(params):
        grad_scale = getattr(param.tag, 'grad_scale', 1)
        if grad_scale != 1:
            grads[idx] *= grad_scale

    
    grads = [lasagne.updates.norm_constraint(grad, GRAD_CLIP, range(grad.ndim))
         for grad in grads]
    
    lr = theano.shared(np.array(LEARNING_RATE,dtype=theano.config.floatX))
    lr_decay = np.array(0.9,dtype=theano.config.floatX)
    updates = lasagne.updates.momentum(grads, params, learning_rate=lr,momentum=0.9)
    #updates = lasagne.updates.adam(grads, params, learning_rate=lr)
    # Theano functions for training and computing cost
    print("Compiling functions ...")
    train = theano.function([input_values, target_values], loss, updates = updates)# allow_input_downcast = True)
    
    probs = theano.function([input_values], network_output)# allow_input_downcast = True)

    read_model_data(net1,'models_s_x3/official/cvprdata_8300.params')
    
        

    # Load the dataset
    print("Loading data...")

    X_train, y_train =  load_CVPR_dataset(3,"/media/titanx2/6B3C546A318EAD3E/DI2K/DIV2K_train_LR_unknown/X3","/media/titanx2/6B3C546A318EAD3E/DI2K/DIV2K_train_HR",60)


    
    # X_train = load_hdf5_data('/media/titanx2/6B3C546A318EAD3E/DI2K/DIV2K_train_LR_unknown/data_x5_48.h5','data')
    # y_train = load_hdf5_data('/media/titanx2/6B3C546A318EAD3E/DI2K/DIV2K_train_LR_unknown/label_x3_48.h5','label')
    print("len y_train: ", len(y_train))
    order = np.random.permutation(len(y_train))
    X_train = X_train[order,:,:,:]
    y_train = y_train[order,:,:,:]

    for i in xrange(10):
        scimc.imsave("low_{}.png".format(i),np.transpose(X_train[i,:,:,:],(1,2,0)))
        scimc.imsave("high_{}.png".format(i),np.transpose(y_train[i,:,:,:],(1,2,0)))
    


        
    print("Training ...")

    def try_an_image(SCALE,data_file,gt_file,result):
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
            
            
        except ValueError:
            print(data_file)
            exit(1)


        
        seq = data_file.rsplit('/')


        scimc.imsave(result+seq[len(seq)-1],super_res)
        return psnr
  
    def batchTest(SCALE,lowImgPath,highImgPath,saveResPath):
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
            
            psnr = try_an_image(SCALE,join(lowImgPath, f), join(highImgPath, gt), saveResPath)
            #continue
            avg_psnr = avg_psnr + psnr        
        avg_psnr = avg_psnr/len(onlyfiles)
        print('OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOH. Average psnr of {}: {}'.format(lowImgPath,avg_psnr))
  
    num_iter = 0
    #batchTest(4,'/media/titanx2/6B3C546A318EAD3E/DI2K/DIV2K_valid_LR_unknown/X4','/media/titanx2/6B3C546A318EAD3E/DI2K/DIV2K_valid_LR_unknown/DIV2K_valid_HR','/media/titanx2/6B3C546A318EAD3E/DI2K/DIV2K_valid_LR_unknown/X4_res/')
    for epoch in range(num_epochs):  
        if (epoch % 5000 == 0 and epoch > 0):
           lr.set_value(lr.get_value()*lr_decay)
                
       # In each epoch, we do a full pass over the training data:

        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, BATCH_SIZE, shuffle=True):           
           if num_iter %200000 == 0 and num_iter>0:
                print("X4")
                #batchTest(3,'./Val/DIV2K_valid_LR_unknown/X3','./DIV2K_valid_HR','./X3_res/')
                #batchTest(4,'/media/titanx2/6B3C546A318EAD3E/DI2K/DIV2K_valid_LR_bicubic/X4','/media/titanx2/6B3C546A318EAD3E/DI2K/DIV2K_valid_LR_bicubic/DIV2K_valid_HR','/media/titanx2/6B3C546A318EAD3E/DI2K/DIV2K_valid_LR_bicubic/X4_res/')
                batchTest(3,'/media/titanx2/6B3C546A318EAD3E/DI2K/DIV2K_valid_LR_unknown/X3','/media/titanx2/6B3C546A318EAD3E/DI2K/DIV2K_valid_LR_unknown/DIV2K_valid_HR','/media/titanx2/6B3C546A318EAD3E/DI2K/DIV2K_valid_LR_unknown/X3_res/')
           if num_iter %100 == 0 and num_iter >=0 :
                batchTest(3,'/media/titanx2/6B3C546A318EAD3E/DI2K/DIV2K_valid_LR_unknown/X3_samples','/media/titanx2/6B3C546A318EAD3E/DI2K/DIV2K_valid_LR_unknown/DIV2K_valid_HR','/media/titanx2/6B3C546A318EAD3E/DI2K/DIV2K_valid_LR_unknown/X3_res/')
           if num_iter %100 == 0 and num_iter>0:

                write_model_data(net1,'models_s_x3/cvprdata_'+str(num_iter) + '.params')                       
                #np.savez('models/MVDSR_'+str(num_iter) +'.npz', *lasagne.layers.get_all_param_values(l_out))



           num_iter = num_iter+1
           inputs, targets = batch
           #print(inputs.shape)         
           err = train(inputs, targets)
           train_err += err
           train_batches += 1

           if train_batches %2 == 0:
           		print("Batch cost: ",err)
           # Then we print the results for this epoch
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))


        
      
			
if __name__ == '__main__':
    main()

