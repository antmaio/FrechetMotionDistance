import math
import random
import numpy as np 

import torch

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from utils import *

class Human36M(Dataset):
    def __init__(self, path, n_poses=34, method=None, std=None, dataset='test'):
        #Init
        frame_stride = 2
        target_joints = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
        self.path = path
        self.n_poses = n_poses 
        self.method=method
        self.std = std
        self.dataset=dataset #['all', 'train', 'test']
        
        if self.dataset == 'train':
            subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9']
        elif self.dataset == 'test':
            subjects = ['S11']
        elif self.dataset == 'all':
            subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
        
        self.motions = []
        self.motion_for_pca = None

        data = np.load(self.path, allow_pickle=True)['positions_3d'].item()
        for subject, actions in data.items():
            if subject not in subjects:
                continue

            for _, positions in actions.items():
                positions = positions[:, target_joints]
                positions = self.normalize(positions)

                for f in range(0, len(positions), 10):
                    if f+n_poses*frame_stride > len(positions):
                        break
                    motion = positions[f:f+n_poses*frame_stride:frame_stride]
                    if method == 'pca_noise':
                        #concatenate all motions for pca fit
                        self.motion_for_pca = np.concatenate((self.motion_for_pca, motion), axis=0) if self.motion_for_pca is not None else motion
                    self.motions.append(motion)
                    
        self.pca_func = PCA().fit(self.motion_for_pca.reshape(len(self.motion_for_pca), -1)) if self.method == 'pca_noise' else None
        
        #stats for minmax rescaling 
        self.mean_norm = []
        self.std_norm = []
        self.bound = {}
        dir_vec = convert_pose_seq_to_dir_vec(np.array(self.motions), 'h36m')
        self.bound['max'] = torch.tensor((dir_vec[..., 0].max(), dir_vec[..., 1].max(), dir_vec[..., 2].max()))
        self.bound['min'] = torch.tensor((dir_vec[..., 0].min(), dir_vec[..., 1].min(), dir_vec[..., 2].min()))
        minmax_dir_vec = (torch.tensor(dir_vec) - self.bound['min']) / (self.bound['max'] - self.bound['min'])

        imagenet_stats=False
        for c in range(3):
            #imagenet_stats 
            if imagenet_stats:
                self.mean_norm = [0.485,0.456,0.406]
                self.std_norm = [0.229, 0.224, 0.225]
            #dataset stats
            else:   
                self.mean_norm.append(minmax_dir_vec[...,c].mean())
                self.std_norm.append(minmax_dir_vec[...,c].std())

            
    def __len__(self):
        return len(self.motions)
    
    def __getitem__(self, index):
        motion = self.motions[index]
        
        #noise to motion
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
                    motion = motion.reshape(len(motion), -1, 3)
                elif len(noise_function(motion, self.std)) == 2: #temporal noise
                    self.noise, r = noise_function(motion, self.std)
                    motion = motion.copy()
                    motion[r:r+self.std] +=  self.noise
                else:
                    motion = motion.copy()
                    self.noise = noise_function(motion, self.std)
                    motion += self.noise
            else:
                raise NotImplementedError
        
        dir_vec = convert_pose_seq_to_dir_vec(motion, 'h36m')
        
        '''
        #MinMax rescaling [-1,1]->[0,1]
        minmax_dir_vec = (torch.tensor(dir_vec) - self.bound['min']) / (self.bound['max'] - self.bound['min'])
        
        #zscore by channel
        
        x_ch0 = (minmax_dir_vec[...,0] - self.mean_norm[0]) / self.std_norm[0]
        x_ch1 = (minmax_dir_vec[...,1] - self.mean_norm[1]) / self.std_norm[1]
        x_ch2 = (minmax_dir_vec[...,2] - self.mean_norm[2]) / self.std_norm[2]
                    
        zdv = torch.cat((x_ch0.unsqueeze(-1), x_ch1.unsqueeze(-1), x_ch2.unsqueeze(-1)), -1)
        '''
        self.zdv = torch.tensor(dir_vec, dtype=torch.float32)
        self.norm_v = torch.permute(self.zdv, (2,1,0))
        
        return torch.from_numpy(motion).float(), self.norm_v
    
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
        noise = np.zeros(data.shape)
        for it in range(len(list_r)):
            noise[list_r[it]] = np.random.normal(0, 0.003**0.5, data[0].shape)
        #noise[np.array(list_r)] = np.random.normal(0,0.003**0.5,data[0].shape)
        return noise

    @staticmethod
    def pca_noise(data, std):
        pass
    
    
    @staticmethod
    def rotation_matrix(axis, theta):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        axis = np.asarray(axis)
        axis = axis / math.sqrt(np.dot(axis, axis))
        a = math.cos(theta / 2.0)
        b, c, d = -axis * math.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
    
    def normalize(self,data):
        '''
        if not self.all_joints:
            hips_idx =  [0,1]
            root_idx = 2 #spine
        else:
            hips_idx = [1, 4]
            root_idx = 0 #central hip
        '''

        hips_idx = [1, 4]
        root_idx = 0 #central hip
        # pose normalization
        for f in range(data.shape[0]):
            data[f, :] -= data[f, root_idx]
            data[f, :, (0, 1, 2)] = data[f, :, (0, 2, 1)]  # xy exchange
            data[f, :, 1] = -data[f, :, 1]  # invert y

        # frontalize based on hip joints
        for f in range(data.shape[0]):
            hip_vec = data[f, hips_idx[1]] - data[f, hips_idx[0]]
            angle = np.pi - np.math.atan2(hip_vec[2], hip_vec[0])  # angles on XZ plane
            if 180 > np.rad2deg(angle) > 0:
                pass
            elif 180 < np.rad2deg(angle) < 360:
                angle = angle - np.deg2rad(360)

            rot = self.rotation_matrix([0, 1, 0], angle)
            data[f] = np.matmul(data[f], rot)

        '''
        if not self.all_joints:
            data = data[:, 2:]  # exclude hip joints
        '''
        return data
    
    @staticmethod
    def rotation_matrix(axis, theta):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        axis = np.asarray(axis)
        axis = axis / math.sqrt(np.dot(axis, axis))
        a = math.cos(theta / 2.0)
        b, c, d = -axis * math.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
