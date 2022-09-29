import warnings
warnings.simplefilter("ignore", UserWarning)

from tqdm import tqdm
import numpy as np
import torch

from fastai.data.all import *
from fastai.vision.all import * 
from fastai.torch_basics import *
from fastai.data.load import *

from utils import *
#from Human36M import Human36M
from TED import TED
from InceptionV3 import InceptionV3


@call_parse
def main(
    all_joints:Param('', bool)=True, 
    n_poses:Param('', int)=34,
    batch_size: Param('', int)=512,
    norm_image: Param('', bool)=True,
    method: Param('', str)='temporal_noise',
    reorg: Param('', bool)=True,
    dataset: Param('[h36m, ted_val, ted_train]', str)='ted_val'
):

    def get_path(ds):
        if ds == 'h36m':
            path  = 'h36m/data_3d_h36m.npz'
        elif 'ted_val' in ds:
            path = 'ted_dataset/lmdb_val/lmdb_val_skel.npz'
        elif 'ted_train' in ds:
            path = 'ted_dataset/lmdb_train/lmdb_train_skel.npz'
        return path

    def get_vecs(dl):
        if len(next(iter(dl))) == 2: #two values to unpack (human36m)
            for i, (_, vec) in enumerate(dl):
                v = vec if i == 0 else torch.cat((v,vec), 0)
        else:
            for i, vec in enumerate(dl):
                v = vec if i == 0 else torch.cat((v,vec), 0)
        return v

    #model 
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx], pretrained=False)
    model = model.cuda()
    model.eval()

    #dataset
    path = '../../Gesture-Generation-from-Trimodal-Context/data/'
    path = path + get_path(dataset)
    mean_dir_vec = np.squeeze(np.array(get_mean_dir_vec(all_joints, dataset)))

    #Imagenet normalization stats
    norm_mean = np.array([0.485,0.456,0.406])
    norm_std = np.array([0.229, 0.224, 0.225])

    ''' 
    ### Noise intensity for ### 
    - salt and pepper noise
    - gaussian noise
    - temporal noise
    - pca noise 
    '''
    #stds = [0.1, 0.15, 0.2, 0.5, 0.75]
    #stds = [0.001, 0.002, 0.003, 0.01, 0.1]
    #stds = [5, 10, 15, 17, 20, 25, 30]
    #stds = [0, 0.1, 0.15, 0.2, 0.5, 1, 2]
    std_ = 17
    std = std_ ** 0.5 if method == 'gaussian_noise' else std_ 

    if 'ted' in dataset:
        
        #Ground truth dataset
        ted_dataset = TED(path, mean_dir_vec, norm_mean_dir_vec=False, to_image=True)
        dataloader = DataLoader(dataset=ted_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        outs = None
        for _, im in tqdm(enumerate(dataloader)):
            out = model(im.cuda()).squeeze()
            outs = torch.cat((outs, out), 0) if outs is not None else out

        n_test = 1 if method == 'pca_noise' else 100
        fgds  = np.empty(n_test)

        for n in range(n_test):

            #Noisy dataset
            ted_dataset_noisy = TED(path, mean_dir_vec, norm_mean_dir_vec=False, to_image=True, method=method, std=std)
            dataloader_noisy = DataLoader(dataset=ted_dataset_noisy, batch_size=batch_size, shuffle=False, drop_last=False)

            outs_noisy = None
            for _, im_noisy in tqdm(enumerate(dataloader_noisy)):
                out_noisy = model(im_noisy.cuda()).squeeze()
                outs_noisy = torch.cat((outs_noisy, out_noisy), 0) if outs_noisy is not None else out_noisy

            print(torch.isnan(outs_noisy).any())
            
            fgd = compute_fgd(outs, outs_noisy)
            fgds[n] = fgd[0]

        #Print fgd 
        print(f'Dataset : {dataset} => fmd between gt and {method} at std of {std_}', fgds.mean(), fgds.std())
    
    else:     
    
        #build datasets and dataloaders
        dataset_ = Human36M(path, mean_dir_vec, n_poses=n_poses, augment=False, all_subject=True, norm_mean=False, all_joints=all_joints)
        valid_dataset_ = Human36M(path, mean_dir_vec, n_poses=n_poses, is_train = False, augment=False, norm_mean=False, all_joints=all_joints)
        
        dataloader_ = DataLoader(dataset=dataset_, batch_size=batch_size, shuffle=True, drop_last=False)
        valid_loader_ = DataLoader(dataset=valid_dataset_, batch_size=batch_size, shuffle=False, drop_last=False)

        #get unit length dir vector
        all_vecs = get_vecs(dataloader_)
        valid_vecs = get_vecs(valid_loader_)
        
        n_joints = all_vecs.shape[2] // 3

        #MinMax rescaling
        all_vecs, valid_vecs = all_vecs.reshape(len(dataset_), n_poses, n_joints, 3), valid_vecs.reshape(len(valid_dataset_), n_poses, n_joints, 3) 
        bounds = Normalization.get_bound_all(all_vecs)

        valid_vecs = (valid_vecs - bounds['min']) / (bounds['max'] - bounds['min'])

        #Create new dataset and dataloaders
        valid_dataset = NormItem(valid_vecs, norm_mean, norm_std, inception=True, reorg=reorg) if norm_image else NormItem(valid_vecs)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        n_test = 1 if method == 'pca_noise' else 100
        fgds = np.empty(n_test)
        for n in range(n_test):

            #Training and save model
            #learn = Learner(dls, model, loss_func = F.l1_loss, metrics=[mae])

            valid_noisy_dataset_ = Human36M(path, mean_dir_vec, n_poses=n_poses, is_train = False, augment=False, norm_mean=False, all_joints=all_joints, method=method, std=std)  #noisy dataset
            valid_noisy_loader_ = DataLoader(dataset=valid_noisy_dataset_, batch_size=batch_size, shuffle=False, drop_last=False)

            valid_noisy_vecs = get_vecs(valid_noisy_loader_)
            valid_noisy_vecs = valid_noisy_vecs.reshape(len(valid_noisy_dataset_), n_poses, n_joints, 3)
            valid_noisy_vecs = (valid_noisy_vecs - bounds['min']) / (bounds['max'] - bounds['min'])

            valid_noisy_dataset = NormItem(valid_noisy_vecs, norm_mean, norm_std, inception=True, reorg=reorg) if norm_image else NormItem(valid_noisy_vecs)
            valid_noisy_loader = DataLoader(dataset=valid_noisy_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        
            outs = None
            for _, (im, _) in tqdm(enumerate(valid_loader)):
                out = model(im.cuda()).squeeze()
                outs = torch.cat((outs, out), 0) if outs is not None else out

            outs_noisy = None
            for _, (im_noisy, _) in tqdm(enumerate(valid_noisy_loader)):
                out_noisy = model(im_noisy.cuda()).squeeze()
                outs_noisy = torch.cat((outs_noisy, out_noisy), 0) if outs_noisy is not None else out_noisy

            fgd = compute_fgd(outs, outs_noisy)
            fgds[n] = fgd[0]
    
        #Print fgd 
        print(f'Dataset : {dataset} => fmd between gt and {method} at std of {std_}', fgds.mean(), fgds.std())
    
