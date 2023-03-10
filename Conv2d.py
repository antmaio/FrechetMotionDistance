from utils import *
from AutoEncoder import *

from fastai.data.all import *
from fastai.vision.all import *
from fastai.torch_basics import *
from fastai.data.load import *

import torch 
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt 
import numpy as np 
from tqdm import tqdm
import os

from dataset.Human36M import Human36M
#from dataset.TED import TED

@call_parse
def main(
    dataset: Param("Dataset to analyze [h36m, ted]", str)='h36m',
    mean_dir_vec:Param("Mean directional vectors", list)=[0.0154009, -0.9690125, -0.0884354, -0.0022264, -0.8655276, 0.4342174, -0.0035145, -0.8755367, -0.4121039, -0.9236511, 0.3061306, -0.0012415, -0.5155854,  0.8129665,  0.0871897, 0.2348464,  0.1846561,  0.8091402,  0.9271948,  0.2960011, -0.013189 ,  0.5233978,  0.8092403,  0.0725451, -0.2037076, 0.1924306,  0.8196916],
    epochs:Param("Number of training epochs", int)=10, 
    batch_size:Param("Training batch size" ,int)=128,
    training:Param("Set to True to train the network with training dataset", bool)=True,
    n_poses:Param("Motion legnth", int)=28,
    method:Param("Type of noise", str)="temporal_noise",
    std: Param('noise standard deviation', float)=0.01,
    strategy: Param("Set to 'gesture' to apply noise on each 34-frames motion. Set to dataset to apply the same noise samples on the whole dataset", str)="gesture",
    norm_image: Param('min max normalize poses', bool)=False,
    all_joints: Param('Set to True to evaluate motion (all body movement). gesture (upper body movement) evaluation otherwise', bool) = False,
    pretrained: Param('', bool)=True
    
): 
    def get_path(ds):
        path = list()
        if ds == 'h36m':
            path.append(os.path.join('h36m', 'data_3d_h36m.npz'))
        elif 'ted' in ds:
            path.append(os.path.join('ted_dataset', 'lmdb_train', 'lmdb_train_skel.npz'))
            path.append(os.path.join('ted_dataset', 'lmdb_val', 'lmdb_val_skel.npz'))
        return path
        
    '''
    def get_vecs(dl):
        for i, (_, vec) in enumerate(dl):
            v = vec if i == 0 else torch.cat((v,vec), 0)
        return v
    '''

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    #model
    model = resnet34(pretrained=pretrained).to(device)
    ae = nn.Sequential(*list(model.children())[:-3])
    conv = nn.Conv2d(256,32,kernel_size=(2,2)).to(device)
    ae.add_module('CodeIn', conv)
    add_layer(ae, 32, 256, 'CodeOut')
    add_layer(ae, 256, 128, 'Upsample0')
    add_layer(ae, 128, 64, 'Upsample1', out_shape=(7,7), scale=None)
    add_layer(ae, 64, 32, 'Upsample2')
    add_layer(ae, 32, 3, 'Upsample3', act='sig')
    if not training:
        loadpth = f'./models/model{n_poses}'
        loadpth = loadpth + '_motion' if all_joints else loadpth
        loadpth = loadpth + '_nopretrained' if not pretrained else loadpth 
        #loadpth = loadpth + '_reorg' if reorg else loadpth
        loadpth = loadpth + '.pth'
        ae.load_state_dict(torch.load(loadpth))

        #Standard deviation 
        std = std ** 0.5 if method == 'gaussian_noise' else std
        std = int(std) if 'temporal' in method else std 


    #Build datasets
    path = get_path(dataset)
    #path = os.path.join('..', 'Gesture-Generation-from-Trimodal-Context', 'data', get_path(dataset))
    if len(path) != 1:
        ted_train_path = os.path.join('..', 'Gesture-Generation-from-Trimodal-Context', 'data', path[0])
        ted_val_path = os.path.join('..', 'Gesture-Generation-from-Trimodal-Context', 'data', path[1])
    else:
        h36m_path = os.path.join('..', 'Gesture-Generation-from-Trimodal-Context', 'data', path[0])

    #Imagenet normalization stats
    #mean_dir_vec = np.squeeze(np.array(mean_dir_vec))
    #norm_mean = np.array([0.485,0.456,0.406])
    #norm_std = np.array([0.229, 0.224, 0.225])


    #dataset
    if 'ted' in dataset:
        ds_func = TED
        train_ds, valid_ds = ds_func(ted_train_path), ds_func(ted_val_path)
        train_loader, valid_loader =  DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=False), DataLoader(valid_ds, batch_size=batch_size, shuffle=False, drop_last=False) 
    elif 'h36m' in dataset:
        ds_func = Human36M
        train_ds, valid_ds = ds_func(h36m_path, dataset='train'), ds_func(h36m_path, dataset='test')
        train_loader, valid_loader = DataLoader(train_ds, batch_size =batch_size, shuffle=False, drop_last=False), DataLoader(valid_ds, batch_size =batch_size, shuffle=False, drop_last=False)
    
    if training: 

        """
        AE Training process 
        """
    
        '''
        #build datasets and dataloaders
        dataset_ = Human36M(path, mean_dir_vec, n_poses=n_poses, augment=False, all_subject=True, norm_mean=False, all_joints=all_joints)
        train_dataset_ = Human36M(path, mean_dir_vec, n_poses=n_poses, is_train=True  , augment=False, norm_mean=False, all_joints=all_joints)  
        valid_dataset_ = Human36M(path, mean_dir_vec, n_poses=n_poses, is_train = False, augment=False, norm_mean=False, all_joints=all_joints)
        
        dataloader_ = DataLoader(dataset=dataset_, batch_size=batch_size, shuffle=True, drop_last=False)
        train_loader_ = DataLoader(dataset=train_dataset_, batch_size=batch_size, shuffle=True, drop_last=False)
        valid_loader_ = DataLoader(dataset=valid_dataset_, batch_size=batch_size, shuffle=False, drop_last=False)

        
        #get unit length dir vector
        all_vecs = get_vecs(dataloader_)
        train_vecs = get_vecs(train_loader_)
        valid_vecs = get_vecs(valid_loader_)

        n_joints = train_vecs.shape[2] // 3

        #MinMax rescaling
        all_vecs, train_vecs, valid_vecs = all_vecs.reshape(len(dataset_), n_poses, n_joints, 3), train_vecs.reshape(len(train_dataset_), n_poses, n_joints, 3), valid_vecs.reshape(len(valid_dataset_), n_poses, n_joints, 3)
        bounds = Normalization.get_bound_all(all_vecs)
        train_vecs, valid_vecs = (train_vecs - bounds['min']) / (bounds['max'] - bounds['min']), (valid_vecs - bounds['min']) / (bounds['max'] - bounds['min'])

        #Create new dataset and dataloaders
        train_dataset = NormItem(train_vecs, norm_mean, norm_std, reorg) if norm_image else NormItem(train_vecs)
        valid_dataset = NormItem(valid_vecs, norm_mean, norm_std, reorg) if norm_image else NormItem(valid_vecs)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        dls = DataLoaders(train_loader, valid_loader) 

        '''
        #Training and save model
        dls = DataLoaders(train_loader, valid_loader)
        learn = Learner(dls, ae, loss_func = F.l1_loss, metrics=[mae])
        suggested_lr = learn.lr_find(stop_div=False, num_it=200)
        print('Training with suggested lr = ', suggested_lr)
        print('Validation loss before fit : ', learn.validate()[0])
        cbs = []
        savepth = f'model{n_poses}'
        savepth = savepth + '_motion' if all_joints else savepth
        savepth = savepth + '_nopretrained' if not pretrained else savepth
        cbs.append(SaveModelCallback(fname=savepth))
        cbs.append(CSVLogger(fname=f'models/log_{savepth}.csv'))
        learn.fit_one_cycle(epochs,lr_max=suggested_lr, cbs=cbs)

    else:

        """
        FGD Measurment process 
        """

        latent_space = []
        
        #hook function to extract the latent space
        def hook(module, input, output):
            latent_space.append(output)
        def hookn(module, input, output):
            latent_space_n.append(output)

        #noise intensity
        '''
        if method == 'gaussian_noise':
            stds = [0.001 ** 0.5, 0.002 ** 0.5, 0.003 ** 0.5, 0.01 ** 0.5, 0.1 ** 0.5]
        elif method == 'saltandpepper_noise': 
            stds = [0.1, 0.15, 0.2, 0.5, 0.75]
        elif method == 'temporal_noise':
            if n_poses == 18:
                stds = [1,5,10,15]
            elif (n_poses == 34) or (n_poses==28):
                stds = [1,5,10,15]
            elif n_poses == 64:
                stds = [1,5,10,15,32]
        '''
        
        """
        valid_dataset_ = Human36M(path, mean_dir_vec, n_poses=n_poses, is_train = False, augment=False, norm_mean=False, all_joints=all_joints)
        valid_loader_ = DataLoader(dataset=valid_dataset_, batch_size=batch_size, shuffle=False, drop_last=False)
        valid_vecs = get_vecs(valid_loader_)

        #MinMax rescaling
        dataset_ = Human36M(path, mean_dir_vec, n_poses=n_poses, augment=False, all_subject=True, norm_mean=False, all_joints=all_joints)
        dataloader_ = DataLoader(dataset=dataset_, batch_size=batch_size, shuffle=True, drop_last=False)
        all_vecs = get_vecs(dataloader_)
        n_joints = valid_vecs.shape[2] // 3
        all_vecs = all_vecs.reshape(len(dataset_), n_poses, n_joints, 3)
        bounds = Normalization.get_bound_all(all_vecs)
        
        valid_vecs = valid_vecs.reshape(len(valid_dataset_), n_poses, n_joints, 3)
        valid_vecs = (valid_vecs - bounds['min']) / (bounds['max'] - bounds['min'])
        
        #Create new dataset and dataloaders with normalized data
        valid_dataset = NormItem(valid_vecs, norm_mean, norm_std, reorg) if norm_image else NormItem(valid_vecs)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        dls = DataLoaders(None, valid_loader)
        """

        dls = DataLoaders(None, valid_loader)
        
        #Compute latent space for ground truth dataset
        learn = Learner(dls, ae, loss_func = F.l1_loss, metrics=[mae])
        handle = learn.model.CodeIn.register_forward_hook(hook)
        _ = learn.get_preds(1)
        handle.remove()
        latent_space = torch.cat(latent_space).detach().cpu().numpy().squeeze()
        
        
        n = 1 if method == 'pca_noise' else 100
        fgds = np.empty(n)

        #Compute latent space for noisy dataset
        for i in tqdm(range(n)):

            latent_space_n = []

            '''
            valid_dataset_noisy_ = Human36M(path, mean_dir_vec, n_poses=n_poses, is_train = False, augment=False, method=method, std=std, norm_mean=False, one_noise_to_all=one_noise_to_all, all_joints=all_joints)
            valid_loader_noisy_ = DataLoader(dataset=valid_dataset_noisy_, batch_size=batch_size, shuffle=False, drop_last=False)
            valid_vecs_noisy = get_vecs(valid_loader_noisy_)

            valid_vecs_noisy = valid_vecs_noisy.reshape(len(valid_dataset_), n_poses, n_joints, 3)
            valid_vecs_noisy = (valid_vecs_noisy - bounds['min']) / (bounds['max'] - bounds['min'])

            valid_dataset_noisy = NormItem(valid_vecs_noisy, norm_mean, norm_std, reorg) if norm_image else NormItem(valid_vecs_noisy)
            valid_loader_noisy = DataLoader(dataset=valid_dataset_noisy, batch_size=batch_size, shuffle=False, drop_last=True)
            '''
            #dataset
            if 'ted' in dataset:
                ds_func = TED
                valid_ds_noisy = ds_func(ted_val_path, method=method, std=std)
                valid_loader_noisy = DataLoader(valid_ds_noisy, batch_size=batch_size, shuffle=False, drop_last=False) 
            elif 'h36m' in dataset:
                ds_func = Human36M
                valid_ds_noisy = ds_func(h36m_path, dataset='test', method=method, std=std)
                valid_loader_noisy = DataLoader(valid_ds_noisy, batch_size =batch_size, shuffle=False, drop_last=False)

            dls_n = DataLoaders(None, valid_loader_noisy)
            learn_n = Learner(dls_n, ae, loss_func = F.l1_loss, metrics=[mae])
            handlen = learn_n.model.CodeIn.register_forward_hook(hookn)
            _ = learn_n.get_preds(1)
            handlen.remove()

            fgd = compute_fgd(latent_space, torch.cat(latent_space_n).detach().cpu().numpy().squeeze())
            fgds[i] = fgd[0]

            #lsn.append(torch.cat(latent_space_n).detach().cpu().numpy().squeeze())
        
            
            """
            for i in range(len(stds)):
                fgd = compute_fgd(latent_space, lsn[i])
                fgds[k,i] = fgd[0]
            """
    
        #Print fmd 
        std_ = std ** 2 if method=='gaussian_noise' else std
        print(f'Dataset : {dataset} | Method : {method} | Stddev : {std_} | from pretrained model : {pretrained} => ', fgds.mean(), fgds.std())
    
    #print(f'fgds with method {method} with n poses = ', n_poses,  fgds.mean(axis=0), fgds.std(axis=0))
    #savepth = f'./evaluation/fgd_{n_poses}_{method}_motion' if all_joints else f'./evaluation/fgd_{n_poses}_{method}'
    #np.savez_compressed(savepth, mean=np.array(fgds.mean(axis=0)), std=np.array(fgds.std(axis=0)))