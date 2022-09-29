from utils import *
from Human36M import Human36M
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

@call_parse
def main(
    variational_encoding: Param("Set to True to enable variational encoding", bool)=False,
    mean_dir_vec:Param("Mean directional vectors", list)=[0.0154009, -0.9690125, -0.0884354, -0.0022264, -0.8655276, 0.4342174, -0.0035145, -0.8755367, -0.4121039, -0.9236511, 0.3061306, -0.0012415, -0.5155854,  0.8129665,  0.0871897, 0.2348464,  0.1846561,  0.8091402,  0.9271948,  0.2960011, -0.013189 ,  0.5233978,  0.8092403,  0.0725451, -0.2037076, 0.1924306,  0.8196916],
    epochs:Param("Number of training epochs", int)=10, 
    batch_size:Param("Training batch size" ,int)=128,
    training:Param("Set to True to train the network with training dataset", bool)=True,
    n_poses:Param("Motion legnth", int)=28,
    method:Param("Type of noise", str)="temporal_noise",
    strategy: Param("Set to 'gesture' to apply noise on each 34-frames motion. Set to dataset to apply the same noise samples on the whole dataset", str)="gesture",
    norm_image: Param('min max normalize poses', bool)=False,
    all_joints: Param('Set to True to evaluate motion (all body movement). gesture (upper body movement) evaluation otherwise', bool) = False,
    reorg: Param('Set to True for the directional vectors reordering to create motion-based image', bool)=True
    
): 
    def get_vecs(dl):
        for i, (_, vec) in enumerate(dl):
            v = vec if i == 0 else torch.cat((v,vec), 0)
        return v
        
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    #nn build and load checkpoint
    model = resnet34(pretrained=True).to(device)
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
        loadpth = loadpth + '_reorg' if reorg else loadpth
        loadpth = loadpth + '.pth'
        ae.load_state_dict(torch.load(loadpth))


    #Build datasets
    path  = '../Gesture-Generation-from-Trimodal-Context/data/h36m/data_3d_h36m.npz'
    mean_dir_vec = np.squeeze(np.array(mean_dir_vec))

    if not variational_encoding:
            loss_func = F.l1_loss

    #Imagenet normalization stats
    norm_mean = np.array([0.485,0.456,0.406])
    norm_std = np.array([0.229, 0.224, 0.225])
  
    if training: 

        """
        AE Training process 
        """

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

        #Training and save model
        learn = Learner(dls, ae, loss_func = loss_func, metrics=[mae])
        suggested_lr = learn.lr_find(stop_div=False, num_it=200)
        print('Training with suggested lr = ', suggested_lr)
        print('Validation loss before fit : ', learn.validate()[0])
        cbs = []
        savepth = f'model{n_poses}'
        savepth = savepth + '_motion' if all_joints else savepth
        savepth = savepth + '_reorg' if reorg else savepth
        cbs.append(SaveModelCallback(fname=savepth))
        cbs.append(CSVLogger(fname='models/log.csv'))
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

        if strategy == 'gesture':
            one_noise_to_all = False
        elif strategy == 'dataset':
            one_noise_to_all = True

        
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
    
        #Compute latent space for ground truth dataset
        learn = Learner(dls, ae, loss_func = loss_func, metrics=[mae])
        handle = learn.model.CodeIn.register_forward_hook(hook)
        _ = learn.get_preds(1)
        handle.remove()
        latent_space = torch.cat(latent_space).detach().cpu().numpy().squeeze()
        
        n = len(valid_dataset) if one_noise_to_all else 10

        fgds = np.empty((n, len(stds)))

        #Compute latent space for noisy dataset
        for k in tqdm(range(n)):
            lsn = []
            for std in stds:
            
                latent_space_n = []

                valid_dataset_noisy_ = Human36M(path, mean_dir_vec, n_poses=n_poses, is_train = False, augment=False, method=method, std=std, norm_mean=False, one_noise_to_all=one_noise_to_all, all_joints=all_joints)
                valid_loader_noisy_ = DataLoader(dataset=valid_dataset_noisy_, batch_size=batch_size, shuffle=False, drop_last=False)
                valid_vecs_noisy = get_vecs(valid_loader_noisy_)

                valid_vecs_noisy = valid_vecs_noisy.reshape(len(valid_dataset_), n_poses, n_joints, 3)
                valid_vecs_noisy = (valid_vecs_noisy - bounds['min']) / (bounds['max'] - bounds['min'])

                valid_dataset_noisy = NormItem(valid_vecs_noisy, norm_mean, norm_std, reorg) if norm_image else NormItem(valid_vecs_noisy)
                valid_loader_noisy = DataLoader(dataset=valid_dataset_noisy, batch_size=batch_size, shuffle=False, drop_last=True)
                
                #Need only the valid dataset but I am not sure if we can use only valid dataset on fastai
                dls_n = DataLoaders(None, valid_loader_noisy)
                learn_n = Learner(dls_n, ae, loss_func = loss_func, metrics=[mae])
                handlen = learn_n.model.CodeIn.register_forward_hook(hookn)
                _ = learn_n.get_preds(1)
                handlen.remove()

                lsn.append(torch.cat(latent_space_n).detach().cpu().numpy().squeeze())
            
            lsn = np.array(lsn)
            
            
            for i in range(len(stds)):
                fgd = compute_fgd(latent_space, lsn[i])
                fgds[k,i] = fgd[0]
        
        print(f'fgds with method {method} with n poses = ', n_poses,  fgds.mean(axis=0), fgds.std(axis=0))
        #savepth = f'./evaluation/fgd_{n_poses}_{method}_motion' if all_joints else f'./evaluation/fgd_{n_poses}_{method}'
        #np.savez_compressed(savepth, mean=np.array(fgds.mean(axis=0)), std=np.array(fgds.std(axis=0)))