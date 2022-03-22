import numpy as np
import random
import matplotlib.pyplot as plt
import math 

import torch
from torch.utils.data import Dataset


from utils import *


train_subject = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9']
test_subject = ['S11']
all_subject = train_subject + test_subject

class Human36M(Dataset):
    """ INIT """
    def __init__(self, path, mean_data, n_poses=34, is_train=True, augment=False, method=None, std=0, to_image=None, one_noise_to_all=False, all_subject=False):
        n_poses = n_poses
        target_joints = [1, 6, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]  # see https://github.com/kenkra/3d-pose-baseline-vmd/wiki/body

        self.method = method
        self.std = std
        self.is_train = is_train
        self.augment = augment
        self.mean_data = mean_data
        self.data = []
        self.one_noise_to_all = one_noise_to_all
        self.noise = None
        
        if all_subject:
            subjects = train_subject + test_subject
        else:
            if is_train:
                subjects = train_subject
            else:
                subjects = test_subject
            
        if self.one_noise_to_all:
            
            if self.method is not None:
                print('One noise sample by dataset')
                if hasattr(self, self.method):
                    noise_function = getattr(self, self.method)
                    noise = noise_function(np.empty((1,10,3)), self.std)
                    self.noise = noise
                    assert len(noise) != 2, "len(noise) = 2" #no temporal noise atm
            else:
                assert False, "There is no noise distribution in method i.e. method is None"
        else:
            if self.method is not None:
                print('One noise sample by gesture')
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
              
    def __getitem__(self, index):
        poses = self.data[index]
        dir_vec = convert_pose_seq_to_dir_vec(poses)
        poses = convert_dir_vec_to_pose(dir_vec)
                
        '''Noise for data augmentation'''
        if self.augment:  # data augmentation by adding gaussian noises on joints coordinates            
            rand_val = random.random()
            if rand_val < 0.2:
                poses = poses.copy()
                poses += np.random.normal(0, 0.002 ** 0.5, poses.shape)
            else:
                poses = poses.copy()
                poses += np.random.normal(0, 0.0001 ** 0.5, poses.shape)
            
        '''Noise for data alteration for FGD sensitivity assessment'''
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
        
        dir_vec = convert_pose_seq_to_dir_vec(poses)
        dir_vec = dir_vec.reshape(dir_vec.shape[0], -1)
        dir_vec = dir_vec - self.mean_data

        poses = torch.from_numpy(poses).float()
        dir_vec = torch.from_numpy(dir_vec).float()
        return poses, dir_vec

    def __len__(self):
        return len(self.data)

    def normalize(self, data):

        # pose normalization
        for f in range(data.shape[0]):
            data[f, :] -= data[f, 2]
            data[f, :, (0, 1, 2)] = data[f, :, (0, 2, 1)]  # xy exchange
            data[f, :, 1] = -data[f, :, 1]  # invert y

        # frontalize based on hip joints
        for f in range(data.shape[0]):
            hip_vec = data[f, 1] - data[f, 0]
            angle = np.pi - np.math.atan2(hip_vec[2], hip_vec[0])  # angles on XZ plane
            if 180 > np.rad2deg(angle) > 0:
                pass
            elif 180 < np.rad2deg(angle) < 360:
                angle = angle - np.deg2rad(360)

            rot = self.rotation_matrix([0, 1, 0], angle)
            data[f] = np.matmul(data[f], rot)

        data = data[:, 2:]  # exclude hip joints
        return data

    @staticmethod
    def to_image(data, name):
        assert len(data.shape) == 3, "data has no valid shape to plot"
        plt.figure()
        plt.imsave(name+'.png', data.numpy())
        
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