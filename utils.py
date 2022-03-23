from lib2to3.refactor import get_all_fix_names
from dataclasses import dataclass
import numpy as np
import random
from scipy import linalg
from scipy.stats import bootstrap
from sklearn.preprocessing import normalize

from fastai.vision.all import *
from fastai.callback.all import *

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch

dir_vec_pairs = [(0, 1, 0.26), (1, 2, 0.18), (2, 3, 0.14), (1, 4, 0.22), (4, 5, 0.36),
                 (5, 6, 0.33), (1, 7, 0.22), (7, 8, 0.36), (8, 9, 0.33)]  # adjacency and bone length


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def convert_dir_vec_to_pose(vec):
    vec = np.array(vec)

    if vec.shape[-1] != 3:
        vec = vec.reshape(vec.shape[:-1] + (-1, 3))

    if len(vec.shape) == 2:
        joint_pos = np.zeros((10, 3))
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[pair[1]] = joint_pos[pair[0]] + pair[2] * vec[j]
    elif len(vec.shape) == 3:
        joint_pos = np.zeros((vec.shape[0], 10, 3))
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[:, pair[1]] = joint_pos[:, pair[0]] + pair[2] * vec[:, j]
    elif len(vec.shape) == 4:  # (batch, seq, 9, 3)
        joint_pos = np.zeros((vec.shape[0], vec.shape[1], 10, 3))
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[:, :, pair[1]] = joint_pos[:, :, pair[0]] + pair[2] * vec[:, :, j]
    else:
        assert False

    return joint_pos


def convert_pose_seq_to_dir_vec(pose):
    if pose.shape[-1] != 3:
        pose = pose.reshape(pose.shape[:-1] + (-1, 3))

    if len(pose.shape) == 3:
        dir_vec = np.zeros((pose.shape[0], len(dir_vec_pairs), 3))
        for i, pair in enumerate(dir_vec_pairs):
            dir_vec[:, i] = pose[:, pair[1]] - pose[:, pair[0]]
            dir_vec[:, i, :] = normalize(dir_vec[:, i, :], axis=1)  # to unit length
    elif len(pose.shape) == 4:  # (batch, seq, ...)
        dir_vec = np.zeros((pose.shape[0], pose.shape[1], len(dir_vec_pairs), 3))
        for i, pair in enumerate(dir_vec_pairs):
            dir_vec[:, :, i] = pose[:, :, pair[1]] - pose[:, :, pair[0]]
        for j in range(dir_vec.shape[0]):  # batch
            for i in range(len(dir_vec_pairs)):
                dir_vec[j, :, i, :] = normalize(dir_vec[j, :, i, :], axis=1)  # to unit length
    else:
        assert False

    return dir_vec


def calculate_latent_space_statistics(ls):
    if type(ls) is not np.ndarray:
        ls = ls.cpu().data.numpy()
    mu = np.mean(ls, axis=0)
    sigma = np.cov(ls, rowvar = False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

#Compute fgd from latent space ls and lsn
def compute_fgd(ls, lsn):
    mu_ls, sigma_ls = calculate_latent_space_statistics(ls)
    mu_lsn, sigma_lsn = calculate_latent_space_statistics(lsn)
    fid = calculate_frechet_distance(mu_ls, sigma_ls, mu_lsn, sigma_lsn, eps=1e-6)
    return fid, mu_ls, mu_lsn, sigma_ls, sigma_lsn


def compute_latent_space(loader, net, variational_encoding, with_noise=False):
    ''' 
    with_noise: If set to True, save noise samples into npy file
    '''
    #generator.pose_encoder.fc_mu.register_forward_hook(get_activation('latent_space'))
    for i, (_,vecs) in enumerate(loader,0):
        target_data = vecs
        #vecs = vecs.reshape(len(vecs), args.n_poses, -1)
        _, context_mu, context_logvar, pose_feat, pose_mu, pose_logvar, recon_data = \
        net(None, None, None, target_data.to(device), variational_encoding=variational_encoding)
        
        if i == 0:
            latent_spaces = pose_feat
            #noises = noise
        else:
            #noises = np.concatenate((noises, noise), axis=0)
            latent_spaces = torch.cat((latent_spaces, pose_feat), 0)
        #lsn = forward(i, vecs.to(device), generator, lsn, args.variational_encoding)
        
    '''
    if with_noise:
        np.savez('noises_one_sample', noises)
    '''
        
    lsn = latent_spaces
    return lsn  


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def bootstrap_fgd(x):
    x = (x,)
    res_std = bootstrap(x, np.std, confidence_level=0.9, random_state = np.random.default_rng(),  method='BCa')
    res_mean = bootstrap(x, np.mean, confidence_level=0.9, random_state = np.random.default_rng(),  method='BCa')
    return res_mean, res_std


class Normalization:
    def get_normalized_all(data):
        #compute min and max on the dataset on all joints
        assert data.shape[-1] == 3, "Last channel is not xyz"
        bound={}
        bound['max'] = torch.tensor((data[..., 0].max(), data[..., 1].max(), data[..., 2].max())) 
        bound['min'] = torch.tensor((data[..., 0].min(), data[..., 1].min(), data[..., 2].min()))

        return (data - bound['min']) / (bound['max'] - bound['min'])
    
    def get_normalized_joint(data):
        assert data.shape[-1] == 3, "Last channel is not xyz"
        data_n = torch.empty(data.shape)
        bound = {}
        
        for i in range(data.shape[-2]):#n joints
            bound[f'max_{i}'] = torch.tensor((data[..., i, 0].max(), data[..., i, 1].max(), data[..., i, 2].max()))
            bound[f'min_{i}'] = torch.tensor((data[..., i, 0].min(), data[..., i, 1].min(), data[..., i, 2].min()))
            
            for o in range(data.shape[-1]):
                data_n[...,i,o] = (data[...,i,o] - bound[f'min_{i}'][o]) / (bound[f'max_{i}'][o] - bound[f'min_{i}'][o])
            
        return data_n
    
    def get_bound_all(data):
        bound = {}
        bound['max'] = torch.tensor((data[..., 0].max(), data[..., 1].max(), data[..., 2].max())) 
        bound['min'] = torch.tensor((data[..., 0].min(), data[..., 1].min(), data[..., 2].min()))
        return bound

    def get_bound_joint(data):
        bound = {}
        for i in range(data.shape[-2]):#n joints
            bound[f'max_{i}'] = torch.tensor((data[..., i, 0].max(), data[..., i, 1].max(), data[..., i, 2].max()))
            bound[f'min_{i}'] = torch.tensor((data[..., i, 0].min(), data[..., i, 1].min(), data[..., i, 2].min()))
        return bound

    def normalize(data, mean, std):
        assert data.shape[-1] == 3, "Last channel is not xyz"
        for c in range(data.shape[-1]):
            data[..., c] = data[..., c] - mean[c] / std[c]
        return data

@dataclass
#Callback to visualize batch as data
class CheckBatch(Callback):
    def __init__(self):
        super().__init__()

    def before_batch(self):
        print(self.x.shape, self.x.dtype)

class NormItem(Dataset):
    def __init__(self,vec):
        self.vec = vec

    def __len__(self):
        return len(self.vec)

    def __getitem__(self, i):
        out = self.resize(self.vec[i], (28,28)).squeeze()
        return out, out

    @staticmethod
    def resize(data, size):
        data = torch.permute(data,(2,1,0)).unsqueeze(0)
        return F.interpolate(data, size, mode='bilinear')
        
    
class Hooks:
    
    def __init__(self):
        self.ls = []

    def hook(self, module, input, output):
        self.ls.append(output)