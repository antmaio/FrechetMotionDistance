import numpy as np
import random
import matplotlib.pyplot as plt
import math 

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from sklearn.decomposition import PCA

from utils import *

train_subject = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9']
test_subject = ['S11']
all_subject = train_subject + test_subject

class Human36M(Dataset):
    """ INIT """
    def __init__(self, path, dataset = 'test', is_pretrained=False, n_poses=34, method=None, std=0, all_joints=True):
        
        #Init
        frame_stride = 2
        #target_joints = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
        self.path = path
        self.n_poses = n_poses 
        self.method=method
        self.std = std
        self.dataset=dataset #['all', 'train', 'test']
        self.all_joints = all_joints
        self.is_pretrained = is_pretrained

        if self.dataset == 'train':
            subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9']
        elif self.dataset == 'test':
            subjects = ['S11']
        elif self.dataset == 'all':
            subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']

        if not all_joints:
            target_joints = [1, 6, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]  # see https://github.com/kenkra/3d-pose-baseline-vmd/wiki/body
        else:
            target_joints = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]  # see https://github.com/kenkra/3d-pose-baseline-vmd/wiki/body    
        '''
        n_poses = n_poses

        self.target_joints = target_joints
        self.method = method
        self.std = std
        self.is_train = is_train
        self.augment = augment
        self.mean_data = mean_data
        self.data = []
        self.one_noise_to_all = one_noise_to_all
        self.noise = None
        self.norm_mean = norm_mean
        self.all_joints = all_joints
        '''

        self.motions = []
        self.motion_for_pca = None

        data = np.load(self.path, allow_pickle=True)['positions_3d'].item()

        n_poses=n_poses
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
        
        #Imagenet stats
        self.norm_mean = np.array([0.485,0.456,0.406])
        self.norm_std = np.array([0.229, 0.224, 0.225])
        self.t = T.Normalize(self.norm_mean, self.norm_std)

        '''    
        if self.one_noise_to_all:
            
            if self.method is not None:
                
                #print('One noise sample by dataset')
                if hasattr(self, self.method):
                    noise_function = getattr(self, self.method)
                    noise = noise_function(np.empty((1,10,3)), self.std)
                    self.noise = noise
                    assert len(noise) != 2, "len(noise) = 2" #no temporal noise atm
            else:
                assert False, "There is no noise distribution in method i.e. method is None"
        else:
            if self.method is not None:
                pass
                #print('One noise sample by gesture')
        # loading data and normalize
        frame_stride = 2
        data = np.load(path, allow_pickle=True)['positions_3d'].item()
        for subject, actions in data.items():
            if subject not in subjects:
                continue

            for action_name, positions in actions.items():
                positions = positions[:, target_joints]
                
                positions = self.normalize(positions)

                for f in range(0, len(positions), 10):
                    if f+n_poses*frame_stride > len(positions):
                        break
                    gesture = positions[f:f+n_poses*frame_stride:frame_stride]
                    self.data.append(gesture)
        '''
              
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
                else:
                    motion = motion.copy()
                    self.noise = noise_function(motion, self.std)
                    motion += self.noise
            else:
                raise NotImplementedError
        
        dir_vec = convert_pose_seq_to_dir_vec(motion, 'h36m')

        """

        if not self.all_joints:
            dir_vec = convert_pose_seq_to_dir_vec(poses,self.all_joints)
            poses = convert_dir_vec_to_pose(dir_vec,self.all_joints)

        """

        '''Noise for data augmentation'''
        '''
        if self.augment:  # data augmentation by adding gaussian noises on joints coordinates            
            rand_val = random.random()
            if rand_val < 0.2:
                poses = poses.copy()
                poses += np.random.normal(0, 0.002 ** 0.5, poses.shape)
            else:
                poses = poses.copy()
                poses += np.random.normal(0, 0.0001 ** 0.5, poses.shape)
        '''

        '''Noise for data alteration for FGD sensitivity assessment'''
        '''
        if self.one_noise_to_all is False:
            if self.method is not None: #poses alteration by adding noises of different types
                if hasattr(self, self.method):
                    noise_function = getattr(self, self.method)
                    if len(noise_function(poses, self.std)) == 2: #temporal noise
                        self.noise, r = noise_function(poses, self.std)
                        poses = poses.copy()
                        poses[r:r+self.std] +=  self.noise
                    else:
                        poses = poses.copy()
                        self.noise = noise_function(poses, self.std)
                        poses += self.noise
        
                else:
                    raise NotImplementedError
        else:
            if self.method is not None:
                poses = poses.copy()
                poses += self.noise
        '''

        self.zdv = torch.tensor(dir_vec, dtype=torch.float32)
        self.norm_v = torch.permute(self.zdv, (2,1,0))

        print(self.norm_v.shape)
        assert False
        size=(28,28)
        self.norm_v=F.interpolate(self.norm_v.unsqueeze(0), size, mode='bilinear', align_corners=True).squeeze()
        
        #dir vec [-1, 1] to [0, 1]
        self.norm_v = (self.norm_v + 1) / 2

        if self.is_pretrained:            
            #imagenet normalization
            self.norm_v = self.t(self.norm_v)

        return self.norm_v, self.norm_v
        #return torch.from_numpy(motion).float(), self.norm_v

    def __len__(self):
        return len(self.motions)

    def normalize(self, data):

        if not self.all_joints:
            hips_idx =  [0,1]
            root_idx = 2 #spine
        else:
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

        if not self.all_joints:
            data = data[:, 2:]  # exclude hip joints

        return data

    @staticmethod
    def to_image(data, name):
        assert len(data.shape) == 3, "data has no valid shape to plot"
        plt.figure()
        plt.imsave(name+'.png', data.numpy())
        
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

        return data
    
