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
#from torch.utils.data import DataLoader

import matplotlib.pyplot as plt 
import numpy as np 
from tqdm import tqdm 

@call_parse
def main(
    name:  Param("name", str)='gesture_autoencoder',
    train_data_path: Param("", str)="data/ted_dataset/lmdb_train",
    val_data_path: Param("", str)="data/ted_dataset/lmdb_val",
    test_data_path: Param("", str)="data/ted_dataset/lmdb_test",
    model_save_path: Param("", str)="output/train_h36m_gesture_autoencoder",
    random_seed: Param("", int)=-1,
    model: Param("", str)='gesture_autoencoder',
    variational_encoding: Param("", bool)=False,
    mean_dir_vec:Param("", list)=[0.0154009, -0.9690125, -0.0884354, -0.0022264, -0.8655276, 0.4342174, -0.0035145, -0.8755367, -0.4121039, -0.9236511, 0.3061306, -0.0012415, -0.5155854,  0.8129665,  0.0871897, 0.2348464,  0.1846561,  0.8091402,  0.9271948,  0.2960011, -0.013189 ,  0.5233978,  0.8092403,  0.0725451, -0.2037076, 0.1924306,  0.8196916],
    mean_pose:Param("", list)= [0.0000306,  0.0004946,  0.0008437,  0.0033759, -0.2051629, -0.0143453,  0.0031566, -0.3054764,  0.0411491,  0.0029072, -0.4254303, -0.001311 , -0.1458413, -0.1505532, -0.0138192, -0.2835603,  0.0670333,  0.0107002, -0.2280813,  0.112117 , 0.2087789,  0.1523502, -0.1521499, -0.0161503,  0.291909 , 0.0644232,  0.0040145,  0.2452035,  0.1115339,  0.2051307],
    epochs:Param("", int)=10, 
    batch_size:Param("" ,int)=128,
    training:Param("set to true to train the network with ground truth data", bool)=False,
    learning_rate: Param("", float)=0.0005,
    motion_resampling_framerate:Param("", int)=15,
    n_poses:Param("", int)=34,
    n_pre_poses:Param("", int)=4, 
    subdivision_stride:Param("", int)=10,
    loader_workers:Param("", int)=4,
    method:Param("Noise distribution", str)="gaussian_noise",
    strategy: Param("How to add noise ?", str)="gesture",
    norm_image: Param('min max normalize poses', bool)=False
    
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
        ae.load_state_dict(torch.load('./models/model.pth'))


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
        dataset_ = Human36M(path, mean_dir_vec, n_poses=n_poses, augment=False, all_subject=True)
        train_dataset_ = Human36M(path, mean_dir_vec, n_poses=n_poses, is_train=True  , augment=False) 
        valid_dataset_ = Human36M(path, mean_dir_vec, n_poses=n_poses, is_train = False, augment=False)
        
        dataloader_ = DataLoader(dataset=dataset_, batch_size=batch_size, shuffle=True, drop_last=False)
        train_loader_ = DataLoader(dataset=train_dataset_, batch_size=batch_size, shuffle=True, drop_last=False)
        valid_loader_ = DataLoader(dataset=valid_dataset_, batch_size=batch_size, shuffle=False, drop_last=False)

        
        #get unit length dir vector
        all_vecs = get_vecs(dataloader_)
        train_vecs = get_vecs(train_loader_)
        valid_vecs = get_vecs(valid_loader_)

        #MinMax rescaling
        all_vecs, train_vecs, valid_vecs = all_vecs.reshape(len(dataset_), n_poses, len(mean_dir_vec) // 3, 3), train_vecs.reshape(len(train_dataset_), n_poses, len(mean_dir_vec) // 3, 3), valid_vecs.reshape(len(valid_dataset_), n_poses, len(mean_dir_vec) // 3, 3)
        bounds = Normalization.get_bound_all(all_vecs)
        train_vecs, valid_vecs = (train_vecs - bounds['min']) / (bounds['max'] - bounds['min']), (valid_vecs - bounds['min']) / (bounds['max'] - bounds['min'])

        #Create new dataset and dataloaders
        train_dataset = NormItem(train_vecs, norm_mean, norm_std) if norm_image else NormItem(train_vecs)
        valid_dataset = NormItem(valid_vecs, norm_mean, norm_std) if norm_image else NormItem(valid_vecs)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        dls = DataLoaders(train_loader, valid_loader)

        #Training and save model
        learn = Learner(dls, ae, loss_func = loss_func, metrics=[mae])
        suggested_lr = learn.lr_find(stop_div=False, num_it=200)
        print('Training with suggested lr = ', suggested_lr)
        print('Validation loss before fit : ', learn.validate()[0])
        cbs = []
        cbs.append(SaveModelCallback())
        cbs.append(CSVLogger(fname='models/log.csv'))
        learn.fit_one_cycle(epochs,lr_max=suggested_lr, cbs=cbs)

        #learn.recorder.plot_losses()
        #learn.show_results(dl_idx=1)

    else:

        """
        FGD Measurment process 
        """

        latent_space = []
        
        def hook(module, input, output):
            latent_space.append(output)
        def hookn(module, input, output):
            latent_space_n.append(output)

        #, 0.002 ** 0.5, 0.003 ** 0.5
        stds = [0.001 ** 0.5, 0.002 ** 0.5, 0.003 ** 0.5, 0.01 ** 0.5, 0.1 ** 0.5]
        #stds = [0.1 ** 0.5]

        if strategy == 'gesture':
            one_noise_to_all = False
        elif strategy == 'dataset':
            one_noise_to_all = True
        
        valid_dataset_ = Human36M(path, mean_dir_vec, n_poses=n_poses, is_train = False, augment=False)
        valid_loader = DataLoader(dataset=valid_dataset_, batch_size=batch_size, shuffle=False, drop_last=True)


        bounds = {}
        print(vecs.max(), vecs.min())

        import pdb; pdb.set_trace()
        '''
        valid_vecs = get_vecs(valid_dataset_)
        if norm_image:
            bounds = Normalization.get_bound_all(get_vecs(dataset))
        else:
            bounds['max'] = 1 
            bounds['min'] = 0

        print(bounds)
        import pdb; pdb.set_trace()

        valid_dataset = NormItem(valid_vecs, bounds)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        '''
        
        dls = DataLoaders(None, valid_loader)
    
        learn = Learner(dls, ae, loss_func = loss_func, metrics=[mae])
        handle = learn.model.CodeIn.register_forward_hook(hook)
        _ = learn.get_preds(1)
        handle.remove()
        latent_space = torch.cat(latent_space).detach().cpu().numpy().squeeze()
        
        n = len(valid_dataset) if one_noise_to_all else 50
        fgds = np.empty((n, len(stds)))
        for k in tqdm(range(n)):
            lsn = []
            for std in stds:
            
                latent_space_n = []
                
                valid_dataset_noisy_ = Human36M(path, mean_dir_vec, n_poses=n_poses, is_train=False, augment=False, method=method, std=std, one_noise_to_all=one_noise_to_all)
                valid_vecs_noisy = get_vecs(valid_dataset_noisy_)
                valid_dataset_noisy = NormItem(valid_vecs_noisy, bounds)
                valid_noisy_loader = DataLoader(dataset=valid_dataset_noisy, batch_size=batch_size, shuffle=False, drop_last=True)
                
                #Need only the valid dataset but I am not sure if we can use only valid dataset on fastai
                dls_n = DataLoaders(None, valid_noisy_loader)
                learn_n = Learner(dls_n, ae, loss_func = loss_func, metrics=[mae])
                handlen = learn_n.model.CodeIn.register_forward_hook(hookn)
                _ = learn_n.get_preds(1)
                handlen.remove()

                lsn.append(torch.cat(latent_space_n).detach().cpu().numpy().squeeze())
            
            lsn = np.array(lsn)
            
            
            for i in range(len(stds)):
                fgd = compute_fgd(latent_space, lsn[i])
                fgds[k,i] = fgd[0]
        
        print(fgds.mean(axis=0), fgds.std(axis=0))



        #print(torch.cat(latent_space).cpu().detach().numpy().squeeze().shape)
        #print(latent_space[0].shape)
        #latent_space = np.array(np.split(np.array(latent_space), len(stds) + 1))
        #print(latent_space.shape)
        

            #batch = dls.valid.one_batch()
            #plt.imshow(batch[0][0].permute(1,2,0))
            #plt.show()
            #out = learn.get_preds(1)

            #print(out[0].shape)

    #print(ae)
    



    '''
    train_dataset = Human36M(path, mean_dir_vec, is_train=True, augment=False)
    val_dataset = Human36M(path, mean_dir_vec, is_train=False, augment=False)
    #val_dataset_noisy = Human36M(path, mean_dir_vec, is_train=False, augment=False, method=method, std=std)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    '''

