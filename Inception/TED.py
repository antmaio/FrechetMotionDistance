import numpy as np
import random
import matplotlib.pyplot as plt
import math 
from sklearn.decomposition import PCA
import torch

from utils import *

class TED(Dataset):
    def __init__(self, path, method=None, std=None):
        self.path = path
        self.method = method
        self.std = std
    
        self.dir_vec = np.load(self.path)['dir_vec']
        self.poses = np.load(self.path)['poses']
        
        self.bound = {}
        self.bound['max'] = torch.tensor((self.dir_vec[..., 0].max(), self.dir_vec[..., 1].max(), self.dir_vec[..., 2].max()))
        self.bound['min'] = torch.tensor((self.dir_vec[..., 0].min(), self.dir_vec[..., 1].min(), self.dir_vec[..., 2].min()))
        
        if self.method == 'pca_noise':
            poses_for_pca = self.poses.reshape(len(self.poses)*self.poses.shape[1], self.poses.shape[2], self.poses.shape[-1])
            self.pca_func = PCA().fit(poses_for_pca.reshape(len(poses_for_pca), -1))
            
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
                    
                elif len(noise_function(motion, self.std)) == 2: #temporal noise
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
        
        dv = convert_pose_seq_to_dir_vec(motion, 'ted')
        dv = torch.tensor(dv, dtype=torch.float32)
        
        #MinMax rescaling [-1,1]->[0,1]
        #self.minmax_dir_vec = (dv - self.bound['min']) / (self.bound['max'] - self.bound['min'])
        self.minmax_dir_vec = torch.permute(dv, (2, 1, 0))
        
        return torch.from_numpy(motion).float(), torch.tensor(self.minmax_dir_vec, dtype=torch.float32)
    
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
    
    @staticmethod
    def temporalV2_noise(data, std):
        list_index = np.arange(0, data.shape[0])
        list_r = []
        for _ in range(std):
            r = np.array(random.choice(list_index))
            list_index = np.setdiff1d(list_index, r)
            list_r.append(r)
        noise=np.zeros(data.shape)
        for it in range(len(list_r)):
            noise[list_r[it]] = np.random.normal(0,0.003**0.5, data[0].shape)
        return noise

    @staticmethod
    def pca_noise(self, data, std):
        pass
