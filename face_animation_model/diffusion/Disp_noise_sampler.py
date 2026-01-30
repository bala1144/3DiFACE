import os
import torch
import torch.nn as nn
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal

class Disp_Noise():
    def __init__(self,
                 dataset_path = None,
                 device="cpu",
                 std_scale=1):

        if dataset_path is None:
            dataset_path = os.path.join(os.getenv("HOME"), "projects/dataset/voca_face_former")

        disp_mean = torch.from_numpy(np.load(os.path.join(dataset_path, "disp_mean.npy"), allow_pickle=True)).float().to(device)
        disp_std = torch.from_numpy(np.load(os.path.join(dataset_path, "disp_std.npy"), allow_pickle=True)).float().to(device)
        print("Loaded displacement noise samplers")

        cov = std_scale * (disp_std + 1e-8) * torch.eye(disp_mean.shape[0]).float().to(device)
        self.distrib = MultivariateNormal(loc=disp_mean, covariance_matrix=cov)

    def sample(self, num_sample):
        return self.distrib.sample((num_sample, ))
    def sample_like_shape(self, shape):
        return self.distrib.sample(shape)
    def sample_like(self, motion):
        shape = motion.shape
        noise = self.sample_like_shape(shape[:-1])
        return noise.to(motion.device)

class random_Noise():

    def __init__(self,
                 dataset_path=None,
                 device="cpu",
                 std_scale=1):
        self.std_scale = std_scale

        if self.std_scale > 0.02:
            print("********************************************************************")
            print("warning using a higher std scale, use 0.02 for standard diff training")
            print("********************************************************************")

    def sample_like(self, motion):
        return torch.rand_like(motion) * self.std_scale
        # return( torch.rand_like(motion) * self.std_scale ) + 1e10
        # return torch.zeros_like(motion) * self.std_scale * 1e-20
    def __call__(self, motion):
        return torch.rand_like(motion) * self.std_scale

class random_Noise_neg_one_2_pos_one():
    def __init__(self,
                 dataset_path=None,
                 device="cpu",
                 std_scale=1):
        self.std_scale = std_scale

    def sample_like(self, motion):
        noise = torch.rand_like(motion)
        noise = (2 * noise) - 1
        return noise * self.std_scale
    
    def __call__(self, motion):
        noise = torch.rand_like(motion)
        noise = (2 * noise) - 1
        return noise * self.std_scale


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        import math
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class random_Noise_pe():
    def __init__(self,
                 dataset_path=None,
                 device="cpu",
                 std_scale=1,
                 latent_dim=15069,
                 ppe_scale=0.001):
        self.std_scale = std_scale
        self.latent_dim = latent_dim
        self.ppe_scale = ppe_scale
        self.PPE = PositionalEncoding(16000, max_len=600)

    def sample_like(self, input):
        """
        input : (Bs x T x num_feat)
        """
        Bs, T, num_feat = input.shape
        # compute the position encoding
        pe = self.PPE.pe[:T].permute(1, 0, 2)[:, :, :self.latent_dim] * self.ppe_scale  # B x T x num_feat
        # sample rand noise
        rand_noise = torch.rand_like(input[:, :1]) * self.std_scale  # B x 1 x num_feat

        return rand_noise + pe.to(input.device)   # 1 x T x num_feat

class Const_random_Noise_pe():
    def __init__(self,
                 dataset_path=None,
                 device="cpu",
                 std_scale=1,
                 latent_dim=15069,
                 ppe_scale=0.001):
        self.std_scale = std_scale
        self.latent_dim = latent_dim
        self.ppe_scale = ppe_scale
        self.PPE = PositionalEncoding(16000, max_len=600)

        self.const_noise = None

    def sample_like(self, input):
        """
        input : (Bs x T x num_feat)
        """

        if self.const_noise is None:
            Bs, T, num_feat = input.shape
            # compute the position encoding
            pe = self.PPE.pe[:T].permute(1, 0, 2)[:, :, :self.latent_dim] * self.ppe_scale  # B x T x num_feat
            # sample rand noise
            rand_noise = torch.rand_like(input[:, :1]) * self.std_scale  # B x 1 x num_feat
            self.const_noise = rand_noise + pe.to(input.device)   # 1 x T x num_feat

        return self.const_noise

class constant_random_Noise():

    def __init__(self,
                 dataset_path=None,
                 device="cpu",
                 std_scale=1):
        self.std_scale = std_scale
        self.const_noise = None

    def sample_like(self, motion):

        """
        motion : B x T X 15069
        """

        if self.const_noise is None:
            print("Sampling Noise for the first time")
            self.const_noise = torch.rand_like(motion) * self.std_scale
            print("First 10 values", self.const_noise[0, 0, :10].cpu().numpy())
            print("Const noise mean", torch.mean(self.const_noise).numpy())

        return self.const_noise.to(motion.device)

class Zero_Noise():
    def __init__(self,
                 dataset_path=None,
                 device="cpu",
                 std_scale=1):
        self.std_scale = std_scale
        self.const_noise = None

    def sample_like(self, motion):
        if self.const_noise is None:
            print("Sampling Noise for the first time")
            self.const_noise = torch.zeros_like(motion) * self.std_scale
        return self.const_noise.to(motion.device)
        # return  torch.zeros_like(motion)

class constant_random_Noise_v2():
    def __init__(self,
                 noise_file,
                 std_scale=1,
                 ):
        self.std_scale = std_scale
        self.const_noise = None

        # noise for the constant experiment
        noise = np.load(os.path.join(os.getenv("HOME"), noise_file))
        self.const_noise = torch.from_numpy(noise).float() * std_scale
        print("noise loaded from", os.path.join(os.getenv("HOME"), noise_file))
        print("scale", self.std_scale)
        print("First 10 values", self.const_noise[0, 0, :10].cpu().numpy())
        print("Const noise mean", torch.mean(self.const_noise).numpy())


    def sample_like(self, motion):
        """
        motion : B x T X 15069
        """
        Bs, T, _ = motion.shape
        return self.const_noise[:Bs, :T].to(motion.device)