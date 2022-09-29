import numpy as np
import random
import matplotlib.pyplot as plt
import math 
from sklearn.decomposition import PCA

import torch
from torch.utils.data import Dataset

from utils import *

class TED(Dataset):

    def __init__(self, path, mean_data, norm_mean_dir_vec, to_image, method=None, std=None) -> None:
        super().__init__()
        
        self.path = path
        self.mean_data = mean_data
        self.norm_mean_dir_vec = norm_mean_dir_vec
        self.to_image = to_image
        self.method = method
        self.std = std

        self.dir_vec = np.load(self.path)['dir_vec']
        self.dir_vec = torch.tensor(self.dir_vec - self.mean_data, dtype=torch.float32) if self.norm_mean_dir_vec else torch.tensor(self.dir_vec, dtype=torch.float32)
        self.poses = np.load(self.path)['poses']

        if self.to_image:
            #min max rescale
            self.bounds = Normalization.get_bound_all(self.dir_vec)

        #Imagenet normalization stats
        self.mean_imagenet = torch.tensor(np.array([0.485,0.456,0.406])).float()
        self.std_imagenet = torch.tensor(np.array([0.229, 0.224, 0.225])).float()

        if self.method == 'pca_noise':
            poses_for_pca = self.poses.reshape(len(self.poses) * self.poses.shape[1] , self.poses.shape[2], self.poses.shape[-1])
            self.pca_func = PCA().fit(poses_for_pca.reshape(len(poses_for_pca), -1))
            #import pdb; pdb.set_trace()

    def __len__(self):
        return len(self.dir_vec)

    def __getitem__(self, i):
 
        motion = self.poses[i]
        
        #Adding noise to motion
        if self.method is not None:
            if hasattr(self, self.method):
                noise_function = getattr(self, self.method)
                
                if self.method == 'pca_noise':
                    motion = motion.copy()
                    #p2pca
                    poses_pca = self.pca_func.transform(motion.reshape(len(motion), -1))
                    poses_pca_n = poses_pca * self.std
                    #pca2p
                    motion = self.pca_func.inverse_transform(poses_pca_n)
                    motion = motion.reshape(len(motion), -1 ,3)

                elif len(noise_function(motion, self.std))== 2: #temporal noise
                    self.noise, r = noise_function(motion, self.std)
                    motion = motion.copy()
                    motion[r:r+self.std] += self.noise
                else:
                    #Gaussian noise or salt and pepper noise
                    self.noise = noise_function(motion, self.std)
                    motion = motion.copy()
                    motion += self.noise
            else:
                raise NotImplementedError

            dv  = convert_pose_seq_to_dir_vec(motion,is_motion=False)
            #if not image normalization, apply mean dir vec substraction
            norm_v = torch.tensor(dv - self.mean_data) if not self.to_image else torch.tensor(dv)
    
        #No noise apply to data (already normalized dir vec)
        else:
            norm_v = self.dir_vec[i]
        
        #Image transformation
        if self.to_image:
            #MinMax rescaling
            dv = (norm_v - self.bounds['min']) / (self.bounds['max'] - self.bounds['min'])
            v = torch.permute(dv, (2,1,0))
            #normalize with imagenet when using pretrained model
            norm_v = torch.tensor((v - self.mean_imagenet[:,None,None]) / self.std_imagenet[:,None,None], dtype=torch.float32)

        return norm_v

    def normalize(self):
        pass

    @staticmethod
    def saltandpepper_noise(data, std):
        u = np.random.uniform(size=data[0].shape) #Applying the same noise on every frame to avoid discontinuities
        noise = np.zeros(u.shape)
        cond0 = np.where(u<=std/2)
        cond1 = np.where((u>std/2) & (u<=std))
        noise[cond0[0], cond0[1]] = 0.2
        noise[cond1[0], cond1[1]] = -0.2
        return noise
    
    @staticmethod
    def gaussian_noise(data, std):
        noise = np.random.normal(0, std, data[0].shape)
        return noise #Applying the same noise on every frame to avoid discontinuities

    @staticmethod
    def temporal_noise(data, std):
        r = np.random.randint(1, data.shape[0] - std - 1)
        noise = np.random.normal(0, 0.003 ** 0.5, data[0].shape)
        return noise, r
    
    def pca_noise(self, data, std):
        
        data = data.copy()
        #p2pca
        poses_pca = self.pca_func.transform(data.reshape(len(data), -1))
        poses_pca_n = poses_pca * self.std
        #pca2p
        poses = self.pca_func.inverse_transform(poses_pca_n)
        return poses.reshape(len(poses), -1 ,3)
