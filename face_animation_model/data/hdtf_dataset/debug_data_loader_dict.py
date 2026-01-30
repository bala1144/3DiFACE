import os
import torch
from collections import defaultdict
from torch.utils import data
import copy
import numpy as np
import pickle
from tqdm import tqdm
import random, math
import glob
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor
import librosa
import itertools

from data.hdtf_dataset.debug_data_loader_faster_collate import DataModuleFromConfig_windows
from torch.utils.data._utils.collate import default_collate
class Dict_data(DataModuleFromConfig_windows):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _train_dataloader(self):
        return data.DataLoader(dataset=self.train_data, batch_size=self.batch_size, shuffle=True, collate_fn=self._colate, num_workers=self.num_workers)

    def _val_dataloader(self):
        return data.DataLoader(dataset=self.valid_data, batch_size=self.batch_size, shuffle=False, collate_fn=self._colate, num_workers=self.num_workers)

    def _test_dataloader(self):
        return data.DataLoader(dataset=self.test_data, batch_size=1, shuffle=False, collate_fn=self._colate, num_workers=self.num_workers)

    def _colate(self, batch):
        """
        batch : is a list of tuples. every row in the batch is seperate sample and every row is
        """
        batch = default_collate(batch)
        audio, vertice, template, one_hot, filename, gt_pose = batch

        if len(audio.shape) == 3:  # handle the cropped case
            bs, t, _ = audio.shape
            audio = audio.reshape(bs * t, -1)
            vertice = vertice.reshape(bs*t, self.window, -1)
            gt_pose = gt_pose.reshape(bs*t, self.window, -1)
            template = template.reshape(bs * t, -1)
            one_hot = one_hot.reshape(bs*t, -1)
            filename = list(itertools.chain(*filename))

        data_dict = {
            "audio":audio,
            "vertice":vertice,
            "template":template,
            "file_name":filename,
            "one_hot":one_hot,
            "pose":gt_pose,
        }
        return data_dict