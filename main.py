from turtle import pd
from utils import *
from fastai.vision.all import *

import torch 
from torch.utils.data import DataLoader

from Human36M import Human36M
from Embedding_net import EmbeddingNet

import matplotlib.pyplot as plt 
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
    epochs:Param("", int)=500, 
    batch_size:Param("" ,int)=128,
    learning_rate: Param("", float)=0.0005,
    motion_resampling_framerate:Param("", int)=15,
    n_poses:Param("", int)=34,
    n_pre_poses:Param("", int)=4, 
    subdivision_stride:Param("", int)=10,
    loader_workers:Param("", int)=4,
    method:Param("Noise distribution", str)="gaussian_noise",
    strategy: Param("How to add noise ?", str)="dataset"
    
): 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pose_dim = 27

    #nn build and load checkpoint
    generator = EmbeddingNet(pose_dim, n_poses, None, None, None, mode='pose').to(device)
    generator.load_state_dict(torch.load(f'../Gesture-Generation-from-Trimodal-Context/output/train_h36m_gesture_autoencoder_noS11_{n_poses}f/gesture_autoencoder_checkpoint_best.bin')['gen_dict'])

    #Build datasets 
    path = '../Gesture-Generation-from-Trimodal-Context/data/h36m/data_3d_h36m.npz'
    mean_dir_vec = np.squeeze(np.array(mean_dir_vec))

    val_dataset = Human36M(path, mean_dir_vec, n_poses=n_poses, is_train=False, augment=False)  
    test_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    #Noise set up
    method_str = method.split('_')[0]
    if method == 'gaussian_noise':
        stds = [0.0316, 0.0447, 0.0548, 0.1, 0.3163]
    elif method == 'saltandpepper_noise':
        stds = [0.1,0.15,0.2] #This is not noise std!
    elif method == 'temporal_noise':
        stds = [1,5,10] #This is not noise std!
    
    #Compute FGD between clean and noisy validation dataset
    latent_spaces_all = []
    latent_spaces_noisy_all = []

    j = 0

    ### Only for testing here : std = 0.1 ** 0.5
    stds = [0.1 ** 0.5]
    for std in stds:
        fgds = []

        if strategy == 'gesture':
            one_noise_to_all = False
        elif strategy == 'dataset':
            one_noise_to_all = True

        val_dataset_noisy = Human36M(path, mean_dir_vec, n_poses=n_poses, is_train=False, augment=False, method=method, std=std, one_noise_to_all=one_noise_to_all)
        #test_loader_noisy = DataLoader(dataset=val_dataset_noisy, batch_size=batch_size, shuffle=False, drop_last=True)
        
        n = len(val_dataset_noisy) if one_noise_to_all else 1

        cov_ls, cov_lsn = np.empty((n*len(stds), 32, 32)), np.empty((n*len(stds), 32, 32))
        mean_ls, mean_lsn = np.empty((n*len(stds), 32)), np.empty((n*len(stds), 32))

        vecs = []

        for _ in range(1):
            val_dataset_noisy = Human36M(path, mean_dir_vec, n_poses=n_poses, is_train=False, augment=False, method=method, std=std, one_noise_to_all=one_noise_to_all)
            test_loader_noisy = DataLoader(dataset=val_dataset_noisy, batch_size=batch_size, shuffle=False, drop_last=True)

            val_dataset = Human36M(path, mean_dir_vec, n_poses=n_poses, is_train=False, augment=False, method=None)
            test_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)


            _, vecn = next(iter(test_loader_noisy))
            _, vec = next(iter(test_loader))
            #for i, (_,_,noise) in enumerate(test_loader_noisy):
                #noises.append(noise)
                #print(noise.mean(), noise.std())
                #import pdb; pdb.set_trace()

            v = vec[0].permute(1,0) 
            vn = vecn[0].permute(1,0)

            print(v.var(), vn.var())
            plt.imshow(np.cov(v), vmin=-0.1, vmax=0.1 )
            #plt.pcolor(X, Y, f(data), vmin=-4, vmax=4)
            plt.colorbar()
            plt.title('gt')

            plt.figure()

            plt.imshow(np.cov(vn),vmin=-0.1, vmax=0.1)
            plt.title('n')
            plt.colorbar()
            plt.show()
            #vecs.append(vec.numpy())
        
        #vecs = np.array(vecs)
        
        '''
        print(vecs.mean(), vecs.std(), vecs.shape, vecs[0,0,0].shape)
        plt.imshow(np.cov(vecs[0,0,0]))
        plt.colorbar()
        plt.show()
        #import pdb; pdb.set_trace()
        '''

        '''
        latent_spaces = compute_latent_space(test_loader, generator, variational_encoding)
        latent_spaces_noisy_gt = compute_latent_space(test_loader_noisy, generator, variational_encoding)
        latent_spaces_all.append(latent_spaces.detach().cpu().numpy())
        latent_spaces_noisy_all.append(latent_spaces_noisy_gt.detach().cpu().numpy())
        
        fgd, mean_ls_, mean_lsn_, cov_ls_, cov_lsn_  = compute_fgd(latent_spaces, latent_spaces_noisy_gt)
        mean_ls[j], mean_lsn[j], cov_ls[j], cov_lsn[j] = mean_ls_, mean_lsn_, cov_ls_, cov_lsn_
        fgds.append(fgd)
        j = j+1
        '''

        fgd_mean, fgd_std = bootstrap_fgd(np.array(fgds))
        print('FGD mean and std for noisy data with psnr of ', std , ':', fgd_mean.confidence_interval, fgd_std.confidence_interval)

    #Saving into npz file 
    save_path = f'stats_{method}_one_sample' if one_noise_to_all else f'stats_{method}'
    np.savez_compressed(save_path, 
                        mean_ls = mean_ls, 
                        mean_lsn = mean_lsn, 
                        cov_ls = cov_ls, 
                        cov_lsn = cov_lsn,
                        latent_spaces = np.array(latent_spaces_all),
                        latent_spaces_noisy = np.array(latent_spaces_noisy_all),
                        fgd = np.array(fgds)
                        #fgd_mean = fgd_mean.confidence_interval, 
                        #fgd_std = fgd_std.confidence_interval
                        )