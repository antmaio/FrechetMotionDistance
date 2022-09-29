import numpy as np 
from sklearn.preprocessing import normalize
from scipy import linalg

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch

def get_dir_vec_pairs(is_motion):
    if not is_motion:
        #ted gesture
        dir_vec_pairs = [(0, 1, 0.26),
                    (1, 2, 0.18), 
                    (2, 3, 0.14), 
                    (1, 4, 0.22), 
                    (4, 5, 0.36),
                    (5, 6, 0.33), 
                    (1, 7, 0.22), 
                    (7, 8, 0.36), 
                    (8, 9, 0.33)]  # adjacency and bone length
    else:
        #human36m motion
        dir_vec_pairs = [(0, 1, 0.136),
            (0, 4, 0.136), 
            (1, 2, 0.449), 
            (2, 3, 0.438), 
            (4, 5, 0.449)  ,
            (5, 6, 0.438), 
            (0, 7, 0.226),
            (7, 8, 0.255), 
            (8, 9, 0.107),
            (9, 10, 0.115),
            (8, 11, 0.139),
            (11, 12, 0.275),
            (12, 13, 0.247),
            (8, 14, 0.139),
            (14, 15, 0.275),
            (15, 16, 0.247)]  # adjacency and bone length

    return dir_vec_pairs

def get_mean_dir_vec(is_motion, dataset=None):
    if dataset == 'h36m' or dataset == None:
        if not is_motion:
            mean_dir_vec = [0.0154009, -0.9690125, -0.0884354, -0.0022264, -0.8655276, 0.4342174, -0.0035145, -0.8755367, -0.4121039, -0.9236511, 0.3061306, -0.0012415, -0.5155854,  0.8129665,  0.0871897, 0.2348464,  0.1846561,  0.8091402,  0.9271948,  0.2960011, -0.013189 ,  0.5233978,  0.8092403,  0.0725451, -0.2037076, 0.1924306,  0.8196916]
        else:
            mean_dir_vec = [9.8999e-01, -5.3267e-02, -2.6510e-10, -9.8999e-01,  5.3267e-02,-2.8980e-10, -7.9123e-03,  7.3447e-01,  3.3218e-01, -5.9390e-02,8.5995e-01, -2.1444e-01, -1.4324e-02,  7.2284e-01,  3.3699e-01,
            7.2409e-02,  8.7285e-01, -1.9341e-01,  1.6671e-02, -9.1132e-01,
            -5.8010e-02,  1.6743e-02, -8.9908e-01,  1.7081e-01,  4.5075e-02,
            -4.9827e-01,  7.1963e-01,  6.1838e-02, -8.3905e-01, -2.7695e-01,
            -9.0950e-01,  3.0537e-01, -7.8299e-02, -4.3069e-01,  7.1239e-01,
            7.7449e-02,  2.4240e-01,  2.8392e-01,  4.3576e-01,  8.9270e-01,
            3.5388e-01, -5.2738e-02,  4.1150e-01,  6.9881e-01,  1.1998e-01,
            -2.2546e-01,  1.3342e-01,  4.4966e-01]
    elif 'ted' in dataset:
        #no body motion for ted, only gesture
        mean_dir_vec = [0.0154009, -0.9690125, -0.0884354, -0.0022264, -0.8655276, 0.4342174, -0.0035145, -0.8755367, -0.4121039, -0.9236511, 0.3061306, -0.0012415, -0.5155854,  0.8129665,  0.0871897, 0.2348464,  0.1846561,  0.8091402,  0.9271948,  0.2960011, -0.013189 ,  0.5233978,  0.8092403,  0.0725451, -0.2037076, 0.1924306,  0.8196916]
    
    return mean_dir_vec
    
def convert_dir_vec_to_pose(vec, is_motion):

    dir_vec_pairs = get_dir_vec_pairs(is_motion)
    vec = np.array(vec)

    if vec.shape[-1] != 3:
        vec = vec.reshape(vec.shape[:-1] + (-1, 3))

    if len(vec.shape) == 2:
        njoints = len(vec) + 1
        joint_pos = np.zeros((njoints, 3))
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[pair[1]] = joint_pos[pair[0]] + pair[2] * vec[j]
    elif len(vec.shape) == 3:
        njoints = vec.shape[1] + 1
        joint_pos = np.zeros((vec.shape[0], njoints, 3))
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[:, pair[1]] = joint_pos[:, pair[0]] + pair[2] * vec[:, j]
    elif len(vec.shape) == 4:  # (batch, seq, ndir, 3)
        njoints = vec.shape[2] + 1
        joint_pos = np.zeros((vec.shape[0], vec.shape[1], njoints, 3))
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[:, :, pair[1]] = joint_pos[:, :, pair[0]] + pair[2] * vec[:, :, j]
    else:
        assert False

    return joint_pos

def convert_pose_seq_to_dir_vec(pose, is_motion):

    dir_vec_pairs = get_dir_vec_pairs(is_motion)
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

''' 
Fréchet distance computation
'''

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

#Compute FMD between latent space ls and lsn
def compute_fgd(ls, lsn):
    mu_ls, sigma_ls = calculate_latent_space_statistics(ls)
    mu_lsn, sigma_lsn = calculate_latent_space_statistics(lsn)
    fid = calculate_frechet_distance(mu_ls, sigma_ls, mu_lsn, sigma_lsn, eps=1e-6)
    return fid, mu_ls, mu_lsn, sigma_ls, sigma_lsn

''' Normalization class '''
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
            data[..., c] = (data[..., c] - mean[c]) / std[c]
        return data

''' Pytorch Dataset '''
class NormItem(Dataset):
    def __init__(self, vec, mean=None, std=None, inception=False, reorg=False):
        self.vec = vec
        self.mean = torch.tensor(mean).float() if mean is not None else None
        self.std = torch.tensor(std).float() if std is not None else None
        self.inception = inception
        self.reorg = reorg
        if self.reorg:
            self.reorg_idx = np.array([0,4,5,4,1,6,7,10,11,12,11,10,8,9,8,13,14,15,14,13,7,6,0,2,3,2])
    def __len__(self):
        return len(self.vec)

    def __getitem__(self, i):
        
        v = self.vec[i][:, self.reorg_idx] if self.reorg else self.vec[i]
 
        resized_data = self.resize(v, (28,28)).squeeze() if not self.inception else torch.permute(v, (2,1,0))
        if self.mean is not None:
            in_data = (resized_data - self.mean[:, None, None]) / self.std[:, None, None]
        else: 
            in_data = resized_data
        out_data = resized_data
        return in_data, out_data

    @staticmethod
    def resize(data, size):
        data = torch.permute(data,(2,1,0)).unsqueeze(0)
        return F.interpolate(data, size, mode='bilinear')