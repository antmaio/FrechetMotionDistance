import warnings 
warnings.simplefilter("ignore", UserWarning)

from tqdm import tqdm
import numpy as np
import os 

import torch
import torch.nn as nn

from fastai.data.all import *
from fastai.vision.all import * 
from fastai.torch_basics import *
from fastai.data.load import *

from utils import *
from Human36M import Human36M
from TED import TED
from InceptionV3 import InceptionV3
from DogLocomotion import DogLocomotion

@call_parse
def main(
    dataset: Param('[h36m, ted_val_, ted_train,dog_locomotion]', str)='h36m',
    model_method : Param('[pretrained, random]', str)='pretrained',
    method: Param('[gaussian_noise, temporal_noise, saltandpepper_noise, pca_noise]', str)='gaussian_noise',
    std: Param('intensity of noise applied to data', float)=0.01,
    batch_size: Param('', int)=128
):
    
    def get_path(ds):
        if ds == 'h36m':
            path = os.path.join('h36m', 'data_3d_h36m.npz')
        elif 'ted_val' in ds:
            path = os.path.join('ted_dataset','lmdb_val','lmdb_val_skel.npz')
        elif 'ted_train' in ds:
            path = os.path.join('ted_dataset', 'lmdb_train', 'lmdb_train_skel.npz')
        elif 'dog' in ds:
            path = 'dog'
        return path

    #model
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    seed = 1
    torch.manual_seed(seed)
    pretrained = True if model_method == 'pretrained' else False
    model = InceptionV3([block_idx], pretrained=pretrained, normalize_input=False)
    model = nn.DataParallel(model).cuda()
    model.eval()

    #dataset
    path = os.path.join('data', get_path(dataset))

    #Standard deviation
    #stds = [0.1, 0.15, 0.2, 0.5, 0.75]
    #stds = [0.001, 0.002, 0.003, 0.01, 0.1]
    #stds = [5, 10, 15, 17, 20, 25, 30]
    #stds = [0, 0.1, 0.15, 0.2, 0.5, 1, 2]
    std = std ** 0.5 if method == 'gaussian_noise' else std
    std = int(std) if 'temporal' in method else std
 
    #### FMD #### 
    if 'ted' in dataset: 
        ds_func = TED
    elif 'dog' in dataset:
        ds_func = DogLocomotion
    elif 'h36m' in dataset:
        ds_func = Human36M
        
    dataset_ = ds_func(path)
    dataloader = DataLoader(dataset_, batch_size=batch_size,shuffle=False, drop_last=False)

    print(next(iter(dataloader))[1].max(), next(iter(dataloader))[1].min())
    #assert False

    #get ground truth data
    outs = None
    for _, (_, dir_vec) in tqdm(enumerate(dataloader)):
        out = model(dir_vec.cuda()).squeeze()
        outs = torch.cat((outs, out), 0) if outs is not None else out

    n_test = 1 if method == 'pca_noise' else 100
    fmds = np.empty(n_test)

    #fmd evaluation
    for n in range(n_test):

        #noisy_dataset
        dataset_ = ds_func(path, method=method, std=std)
        dataloader = DataLoader(dataset_, batch_size=batch_size, shuffle=False, drop_last = False)

        outs_noisy = None
        for _, (_, dir_vec_noisy) in tqdm(enumerate(dataloader)):
            out_noisy = model(dir_vec_noisy.cuda()).squeeze()
            outs_noisy = torch.cat((outs_noisy, out_noisy), 0) if outs_noisy is not None else out_noisy

        print(torch.isnan(outs_noisy).any())

        fmd = compute_fgd(outs, outs_noisy)
        fmds[n] = fmd[0]

    #Print fgd 
    std_ = std ** 2 if method=='gaussian_noise' else std
    print(f'Dataset : {dataset} => fmd between gt and {method} at std of {std_}', fmds.mean(), fmds.std())
    
     

