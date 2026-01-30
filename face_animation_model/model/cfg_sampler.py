import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

# A wrapper model for Classifier-free guidance **SAMPLING** only
# https://arxiv.org/abs/2207.12598
class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model, scale_factor):
        super().__init__()
        self.model = model  # model is the actual model to run
        self.guidance_factor = scale_factor  # model is the actual model to run
        assert self.model.cond_mask_prob > 0, 'Cannot run a guided diffusion on a model that has not been trained with no conditions'

    def forward(self, x, timesteps, **kwargs):
        cond_mode = self.model.cond_mode
        disp_out_cond = self.model(x, timesteps, **kwargs)

        # generate unconditional motion
        self.model.cond_mode = "no_cond"
        disp_out_uncond = self.model(x, timesteps, **kwargs)

        # reset the condition
        self.model.cond_mode = cond_mode

        return disp_out_uncond + (self.guidance_factor * (disp_out_cond - disp_out_uncond))