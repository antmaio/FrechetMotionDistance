import torch.nn as nn
from utils import *
import random

def ConvNormRelu(in_channels, out_channels, downsample=False, padding=0, batchnorm=True):
    if not downsample:
        k = 3
        s = 1
    else:
        k = 4
        s = 2

    conv_block = nn.Conv1d(in_channels, out_channels, kernel_size=k, stride=s, padding=padding)
    norm_block = nn.BatchNorm1d(out_channels)

    if batchnorm:
        net = nn.Sequential(
            conv_block,
            norm_block,
            nn.LeakyReLU(0.2, True)
        )
    else:
        net = nn.Sequential(
            conv_block,
            nn.LeakyReLU(0.2, True)
        )

    return net


class PoseEncoderConv(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.net = nn.Sequential(
            ConvNormRelu(dim, 32, batchnorm=True),
            ConvNormRelu(32, 64, batchnorm=True),
            ConvNormRelu(64, 64, True, batchnorm=True),
            nn.Conv1d(64, 32, 3)
        )

        """ for 34 or 64 frames """
        self.out_net = nn.Sequential(
            #nn.Linear(864, 256),  # for 64 frames
            nn.Linear(384, 256),  # for 34 frames
            nn.BatchNorm1d(256),
            nn.LeakyReLU(True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(True),
            nn.Linear(128, 32),
        )
        
        """ for 18 frames """
        '''
        self.out_net = nn.Sequential(
            nn.Linear(128, 64), 
            nn.BatchNorm1d(64),
            nn.LeakyReLU(True),
            nn.Linear(64, 32)
        )
        '''

        self.fc_mu = nn.Linear(32, 32)
        self.fc_logvar = nn.Linear(32, 32)

    def forward(self, poses, variational_encoding):
        # encode
        poses = poses.transpose(1, 2)  # to (bs, dim, seq)
        out = self.net(poses)
        out = out.flatten(1)
        out = self.out_net(out)

        # return out, None, None
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)

        if variational_encoding:
            z = reparameterize(mu, logvar)
        else:
            z = mu
        return z, mu, logvar
    
class PoseDecoderConv(nn.Module):
    def __init__(self, length, dim, use_pre_poses=False):
        super().__init__()
        self.use_pre_poses = use_pre_poses

        feat_size = 32
        
        if use_pre_poses:
            self.pre_pose_net = nn.Sequential(
                nn.Linear(dim * 4, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Linear(32, 32),
            )
            feat_size += 32

        if length == 64:
            self.pre_net = nn.Sequential(
                nn.Linear(feat_size, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(True),
                nn.Linear(128, 256),
            )
        elif length == 34:
            self.pre_net = nn.Sequential(
                nn.Linear(feat_size, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(True),
                nn.Linear(64, 136),
            )
        elif length == 18:
            self.pre_net = nn.Sequential(
                nn.BatchNorm1d(feat_size),
                nn.LeakyReLU(True),
                nn.Linear(feat_size, 72),
            )
        
        else:
            assert False

        self.net = nn.Sequential(
            nn.ConvTranspose1d(4, 32, 3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose1d(32, 32, 3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(32, 32, 3),
            nn.Conv1d(32, dim, 3),
        )

    def forward(self, feat, pre_poses=None):
        if self.use_pre_poses:
            pre_pose_feat = self.pre_pose_net(pre_poses.reshape(pre_poses.shape[0], -1))
            feat = torch.cat((pre_pose_feat, feat), dim=1)

        out = self.pre_net(feat)
        out = out.view(feat.shape[0], 4, -1)
        out = self.net(out)
        out = out.transpose(1, 2)
        return out
    
class EmbeddingNet(nn.Module):
    def __init__(self, pose_dim, n_frames, n_words, word_embed_size, word_embeddings, mode):
        super().__init__()
        if mode != 'pose':
            raise NotImplementedError 
            #self.context_encoder = ContextEncoder(args, n_frames, n_words, word_embed_size, word_embeddings)
            #self.pose_encoder = PoseEncoderConv(n_frames, pose_dim)
            #self.decoder = PoseDecoderFC(n_frames, pose_dim, use_pre_poses=True)
            #self.decoder = PoseDecoderGRU(n_frames, pose_dim)
        else:
            self.context_encoder = None
            self.pose_encoder = PoseEncoderConv(n_frames, pose_dim)
            self.decoder = PoseDecoderConv(n_frames, pose_dim)
        self.mode = mode

    def forward(self, in_text, in_audio, pre_poses, poses, input_mode=None, variational_encoding=False):
        if input_mode is None:
            assert self.mode is not None
            input_mode = self.mode

        # context
        if self.context_encoder is not None and in_text is not None and in_audio is not None:
            context_feat, context_mu, context_logvar = self.context_encoder(in_text, in_audio)
            # context_feat = F.normalize(context_feat, p=2, dim=1)
        else:
            context_feat = context_mu = context_logvar = None

        # poses
        if poses is not None:
            poses_feat, pose_mu, pose_logvar = self.pose_encoder(poses, variational_encoding)
            # poses_feat = F.normalize(poses_feat, p=2, dim=1)
        else:
            poses_feat = pose_mu = pose_logvar = None

        # decoder
        if input_mode == 'random':
            input_mode = 'speech' if random.random() > 0.5 else 'pose'

        if input_mode == 'speech':
            latent_feat = context_feat
        elif input_mode == 'pose':
            latent_feat = poses_feat
        else:
            assert False

        out_poses = self.decoder(latent_feat, pre_poses)

        return context_feat, context_mu, context_logvar, poses_feat, pose_mu, pose_logvar, out_poses

    def freeze_pose_nets(self):
        for param in self.pose_encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False