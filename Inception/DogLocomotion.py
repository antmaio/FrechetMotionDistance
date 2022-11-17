import torch
from torch.utils.data import Dataset

import numpy as np 
import os
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

from pymo.parsers import BVHParser
from pymo.preprocessing import *
from pymo.viz_tools import *

from utils import *

class DogLocomotion(Dataset):
    def __init__(self, path, nframes=34, method=None, std=None, to_image=True):
        self.path = path
        self.method = method
        self.std = std
        self.to_image = to_image
        self.nframes = nframes
        
        #skeleton structure 
        dog_bones = [
         #Hips->Tail
         (0,1),
         (1,2),
         (2,3),
         #Hips->RightLeg
         (0,4),
         (4,5),
         (5,6),
         (6,7),
         #Hips->LeftLeg 
         (0,8),
         (8,9),
         (9,10),
         (10,11),
         #Hips->Spine
         (0,12),
         (12,13),
         #Spine->RightHand
         (13,14),
         (14,15),
         (15,16),
         (16,17),
         (17,18),
         #Spine->LeftHand
         (13,19),
         (19,20),
         (20,21),
         (21,22),
         (22,23),
         #Spine->Head
         (13,24),
         (24,25),
         (25,26)]
        
        #List all bvh files in directory
        bvh_files = [f for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f))]
        
        #BVH preprocessing
        data_pipe = Pipeline([
            ('param', MocapParameterizer('position')),
            ('np', Numpyfier())
        ])
        
        #Extract Cartesian positions from bvh
        parser = BVHParser()
        positions = []
        for bvh_file in bvh_files:
            bvh_data = parser.parse(os.path.join(self.path,bvh_file))
            position_ = data_pipe.fit_transform([bvh_data]).squeeze()
            positions.append(position_)
        
        #Dataset preparation
        data = []
        frame_stride = 2 #Downsampling 
        for i in range(len(positions)):
            pos = positions[i]
            for f in range(0, len(pos)):
                if f+self.nframes*frame_stride > len(pos):
                    break
                motion = pos[f:f+self.nframes*frame_stride:frame_stride]
                data.append(motion)

        self.data = np.array(data).reshape(len(data), self.nframes, position_.shape[1]//3, 3)
        
        #compute pca_func on positions if pca_noise
        if self.method == 'pca_noise':
            poses_for_pca = self.data.reshape(len(self.data) * self.data.shape[1], 
                                              self.data.shape[2], self.data.shape[-1])
            self.pca_func = PCA().fit(poses_for_pca.reshape(len(poses_for_pca), -1))
        
        #Extract norm of each bones
        self.norm_bones = []
        for bone in dog_bones:
            norm_bone = np.linalg.norm(self.data[0,bone[1],0]-self.data[0,bone[0],0])
            self.norm_bones.append(norm_bone)
        
        #Pos2dirvec
        self.dir_vec = convert_pose_seq_to_dir_vec(self.data, 'dog', val=self.norm_bones) 
        
        if self.to_image:
            
            #stats for MinMax rescaling
            self.bound = {}
            self.bound['max'] = torch.tensor((self.dir_vec[..., 0].max(), self.dir_vec[..., 1].max(), self.dir_vec[..., 2].max()))
            self.bound['min'] = torch.tensor((self.dir_vec[..., 0].min(), self.dir_vec[..., 1].min(), self.dir_vec[..., 2].min()))
            
            minmax_dir_vec = (torch.tensor(self.dir_vec) - self.bound['min']) / (self.bound['max'] - self.bound['min'])
            
            '''
            self.mean = []
            self.std = []
            
            for c in range(3):
                self.mean.append(minmax_dir_vec[...,c].mean())
                self.std.append(minmax_dir_vec[...,c].std())
            '''
            
    def __len__(self):
        return len(self.dir_vec) 
    
    def __getitem__(self,i):
        
        motion = self.data[i]
        
        #Add noise to motion
        if self.method is not None:
            if hasattr(self, self.method):
                noise_function = getattr(self, self.method)
        
                if self.method == 'pca_noise':
                    motion = motion.copy()
                    #p2pca
                    #import pdb; pdb.set_trace()
                    poses_pca = self.pca_func.transform(motion.reshape(len(motion), -1))
                    poses_pca_n = poses_pca * self.std
                    #pca2p
                    motion = self.pca_func.inverse_transform(poses_pca_n)
                    motion = motion.reshape(len(motion), -1 ,3)
                     
                elif len(noise_function(motion, self.std))==2:
                    self.noise, r = noise_function(motion, self.std)
                    motion = motion.copy()
                    motion[r:r+self.std] += self.noise
                else: 
                    #spatial noise
                    self.noise = noise_function(motion, self.std)
                    motion = motion.copy()
                    motion += self.noise
                    
            #method is not found in Locomotion
            else:
                raise NotImplementedError
                
        #No noise applied to data 
        dir_vec = self.dir_vec[i] if self.method is None else convert_pose_seq_to_dir_vec(motion, 'dog', val=self.norm_bones) 

        if self.to_image:
            '''
            #MinMax rescaling [-1, 1] -> [0, 1]
            minmax_dir_vec = (torch.tensor(dir_vec) - self.bound['min']) / (self.bound['max'] - self.bound['min'])
            
            #zscore by channel
            
            x_ch0 = (minmax_dir_vec[...,0] - self.mean[0]) / self.std[0]
            x_ch1 = (minmax_dir_vec[...,1] - self.mean[1]) / self.std[1]
            x_ch2 = (minmax_dir_vec[...,2] - self.mean[2]) / self.std[2]
            
            zdv = torch.cat((x_ch0.unsqueeze(-1), x_ch1.unsqueeze(-1), x_ch2.unsqueeze(-1)), -1)
            '''
            self.zdv = torch.tensor(dir_vec, dtype=torch.float32)
            self.norm_v = torch.permute(self.zdv, (2,1,0))
            
        return torch.from_numpy(motion).float(), torch.tensor(self.norm_v, dtype=torch.float32)
    
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
    def pca_noise(data, std):
        pass
    
