import warnings 
warnings.simplefilter("ignore", UserWarning)

from tqdm import tqdm
import numpy as np
import os 
import scipy.stats

import torch
import torch.nn as nn
from torch.nn.functional import adaptive_avg_pool2d

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
    batch_size: Param('', int)=128,
    n_channels:Param('# of kept channels (from most discr to less discr.)', int)=0,
    bidx: Param('channel feature map [64, 192, 768, 2048]',int)=2048
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

    def get_stddev(method):
        if method=='gaussian_noise':
            stddev = [0.001,0.002,0.003,0.01,0.1]
        elif method == 'saltandpepper_noise':
            stddev = [0.1,0.15,0.2,0.5,0.75]
        elif method == 'temporalV2_noise':
            stddev = [5,10,15,30]
        elif method == 'pca_noise':
            stddev = [0,0.1,0.15,0.2,0.5,2]
        return stddev

    #Keep most informative channels
    if n_channels != 0:
        distance = np.load(f'distance_{bidx}_{dataset}.npz')
        d_ = np.zeros((len(get_stddev(method)),bidx))
        for ids,stdd in enumerate(get_stddev(method)):
            #print(distance[f'{method}_{stdd}'])
            d_[ids] = distance[f'{method}_{stdd}']
        sorted_channels=np.array(scipy.stats.mode(d_,axis=0).mode[0], dtype=np.int64)
        #print('resulting vector')
        print(sorted_channels)
    else:
        sorted_channels=None
    

    #model
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[bidx]
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
        out = adaptive_avg_pool2d(out, output_size=(1,1))
        outs = torch.cat((outs, out), 0) if outs is not None else out

    outs_cpu = torch.tensor(outs.detach().cpu(), dtype=torch.float32)
    outs_cpu = outs_cpu[:,sorted_channels[-n_channels:]] if sorted_channels is not None else outs_cpu

    del outs #to free memory

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
            if bidx !=2048:
                out_noisy = adaptive_avg_pool2d(out_noisy, output_size=(1,1))
            outs_noisy = torch.cat((outs_noisy, out_noisy), 0) if outs_noisy is not None else out_noisy

        outs_noisy_cpu = torch.tensor(outs_noisy.detach().cpu(),dtype=torch.float32)
        del outs_noisy

        outs_noisy_cpu = outs_noisy_cpu[:,sorted_channels[-n_channels:]] if sorted_channels is not None  else outs_noisy_cpu

        fmd = compute_fgd(outs_cpu.squeeze(), outs_noisy_cpu.squeeze())
        fmds[n] = fmd[0]

    #Print fgd 
    std_ = std ** 2 if method=='gaussian_noise' else std
    print(f'Dataset : {dataset} | Method : {method} | Stddev : {std_} | Number of kept channels : {n_channels} | Emebdding size : {bidx} => ', fmds.mean(), fmds.std())
    
    

