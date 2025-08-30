import os
import torch
from collections import defaultdict
from torch.utils import data
import copy
import numpy as np
import pickle
from tqdm import tqdm
import random,math
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor
import librosa

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data,subjects_dict,data_type="train"):
        self.data = data
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type
        self.one_hot_labels = np.eye(len(subjects_dict["train"]))

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        file_name = self.data[index]["name"]
        audio = self.data[index]["audio"]
        vertice = self.data[index]["vertice"]
        template = self.data[index]["template"]
        if self.data_type == "train":
            subject = "_".join(file_name.split("_")[:-1])
            one_hot = self.one_hot_labels[self.subjects_dict["train"].index(subject)]
        else:
            one_hot = self.one_hot_labels
        return torch.FloatTensor(audio),torch.FloatTensor(vertice), torch.FloatTensor(template), torch.FloatTensor(one_hot), file_name

    def __len__(self):
        return self.len

def read_data(
        dataset,
        dataset_root,
        wav_path,
        vertices_path,
        template_file,
        train_subjects,
        val_subjects,
        test_subjects,
        **kwargs
              ):
    print("Loading data...")
    data = defaultdict(dict)
    train_data = []
    valid_data = []
    test_data = []

    audio_path = os.path.join(os.getenv("DATA_HOME"), dataset_root, wav_path)
    vertices_path = os.path.join(os.getenv("DATA_HOME"), dataset_root, vertices_path)
    wav2vec_path = os.path.join(os.getenv('HOME'), dataset_root, "wav2vec2-base-960h")
    if not os.path.isdir(wav2vec_path):
        print("Using global processor to process the model")
        wav2vec_path = "facebook/wav2vec2-base-960h"
    else:
        print("using local processor")
    processor = Wav2Vec2Processor.from_pretrained(wav2vec_path)

    template_file = os.path.join(os.getenv("DATA_HOME"), dataset_root, template_file)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin,encoding='latin1')
    
    for r, ds, fs in os.walk(audio_path):
        for f in tqdm(fs):
            if f.endswith("wav"):
                wav_path = os.path.join(r,f)
                speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
                input_values = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
                key = f.replace("wav", "npy")
                data[key]["audio"] = input_values
                subject_id = "_".join(key.split("_")[:-1])
                temp = templates[subject_id]
                data[key]["name"] = f
                data[key]["template"] = temp.reshape((-1)) 
                vertice_path = os.path.join(vertices_path,f.replace("wav", "npy"))
                if not os.path.exists(vertice_path):
                    del data[key]
                else:
                    if dataset == "vocaset":
                        data[key]["vertice"] = np.load(vertice_path,allow_pickle=True)[::2,:]#due to the memory limit
                    elif dataset == "BIWI":
                        data[key]["vertice"] = np.load(vertice_path,allow_pickle=True)

                # # this is just for debugging the local machine
                # if len(data) > 100:
                #     break

    subjects_dict = {}
    subjects_dict["train"] = [i for i in train_subjects.split(" ")]
    subjects_dict["val"] = [i for i in val_subjects.split(" ")]
    subjects_dict["test"] = [i for i in test_subjects.split(" ")]

    splits = {'vocaset':{'train':range(1,41),'val':range(21,41),'test':range(21,41)},
     'BIWI':{'train':range(1,33),'val':range(33,37),'test':range(37,41)}}
   
    for k, v in data.items():
        subject_id = "_".join(k.split("_")[:-1])
        sentence_id = int(k.split(".")[0][-2:])
        if subject_id in subjects_dict["train"] and sentence_id in splits[dataset]['train']:
            train_data.append(v)
        if subject_id in subjects_dict["val"] and sentence_id in splits[dataset]['val']:
            valid_data.append(v)
        if subject_id in subjects_dict["test"] and sentence_id in splits[dataset]['test']:
            test_data.append(v)

    print(len(train_data), len(valid_data), len(test_data))
    return train_data, valid_data, test_data, subjects_dict

def get_dataloaders(args):
    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_data(args)
    train_data = Dataset(train_data,subjects_dict,"train")
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    valid_data = Dataset(valid_data,subjects_dict,"val")
    dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=False)
    test_data = Dataset(test_data,subjects_dict,"test")
    dataset["test"] = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    return dataset



class DataModuleFromConfig():
    def __init__(self, **kwargs):
        super().__init__()
        train_data, valid_data, test_data, subjects_dict = read_data(**kwargs)
        self.train_data = Dataset(train_data, subjects_dict, "train")
        self.valid_data = Dataset(valid_data, subjects_dict, "val")
        self.test_data = Dataset(test_data, subjects_dict, "test")

        # need for the pytorhch lightning modeule to work
        self.dataset_configs = {}
        self.dataset_configs["train"] = self.train_data
        self.train_dataloader = self._train_dataloader
        self.dataset_configs["validation"] = self.valid_data
        self.val_dataloader = self._val_dataloader
        self.dataset_configs["test"] = self.test_data
        self.test_dataloader = self._test_dataloader

    def prepare_data(self):
        print("prepere data do nothin")

    def setup(self, stage=None):
        print("setup do nothin")

    def _train_dataloader(self):
        return data.DataLoader(dataset=self.train_data, batch_size=1, shuffle=True)

    def _val_dataloader(self):
        return data.DataLoader(dataset=self.valid_data, batch_size=1, shuffle=True)

    def _test_dataloader(self):
        return data.DataLoader(dataset=self.test_data, batch_size=1, shuffle=True)

    def _test_unseen_dataloader(self):
        # Need to this place holder function for the tester
        return None


from torch.utils.data._utils.collate import default_collate
class DataModuleFromConfig_fixed_windows(DataModuleFromConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = kwargs.get("batch_size", 1)
        self.window = kwargs.get("window", 30)
        self.fps = kwargs.get("fps", 30)
        self.audio_sample_rate = kwargs.get("audio_sample_rate", 16000)
        self.num_patches_per_seq = kwargs.get("num_patches_per_seq", 5)
        self.num_workers = kwargs.get("num_workers", 0)
        print("Number of workers", self.num_workers)


    def _train_dataloader(self):
        return data.DataLoader(dataset=self.train_data, batch_size=self.batch_size, shuffle=True, collate_fn=self._colate, num_workers=self.num_workers)

    def _val_dataloader(self):
        return data.DataLoader(dataset=self.valid_data, batch_size=1, shuffle=False, collate_fn=self._colate, num_workers=self.num_workers)

    def _test_dataloader(self):
        return data.DataLoader(dataset=self.test_data, batch_size=1, shuffle=False, collate_fn=self._colate, num_workers=self.num_workers)

    def _colate(self, batch):
        """
        batch : is a list of tuples. every row in the batch is seperate sample and every row is
        """

        new_batch = []
        for full_sample in batch:
            audio, vertice, template, one_hot, file_name = full_sample
            seq_len = vertice.shape[0]
            if self.window < seq_len:
                # choose a random starting point for the seqs
                possible_idxs = np.arange(0, seq_len - self.window)
                np.random.shuffle(possible_idxs)
                num_of_samples = min(len(possible_idxs), self.num_patches_per_seq)

                for id in range(num_of_samples):
                    start_idx = possible_idxs[id]
                    end_idx = start_idx + self.window
                    sampled_vertice = vertice[start_idx:end_idx]
                    # sample the audio accordingly
                    audio_start = int((start_idx / self.fps) * self.audio_sample_rate)
                    audio_end = int((end_idx / self.fps) * self.audio_sample_rate)
                    sampled_audio = audio[audio_start:audio_end]

                    new_file_name = file_name.replace(".wav", "_start%03d.wav"%start_idx)
                    new_batch.append((sampled_audio, sampled_vertice, template, one_hot, new_file_name))
            else:
                seq_len = (seq_len // 8) * 8  # to enable compression
                sampled_vertice = vertice[:seq_len]
                audio_end = int((seq_len / self.fps) * self.audio_sample_rate)
                sampled_audio = audio[:audio_end]
                new_batch.append((sampled_audio, sampled_vertice, template, one_hot, file_name))

        # using the default collate to group the fields
        return default_collate(new_batch)

# #######################################################################################################
# # Adeed on Jul17 for conditional and unconditional combined experiment
# ######################################################################################################
#
# class DataModuleFromConfig_fixed_windows_random_uncond(DataModuleFromConfig):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.batch_size = kwargs.get("batch_size", 1)
#         self.window = kwargs.get("window", 30)
#         self.fps = kwargs.get("fps", 30)
#
#         self.audio_sample_rate = kwargs.get("audio_sample_rate", 16000)
#         self.num_patches_per_seq = kwargs.get("num_patches_per_seq", 5)
#         self.num_workers = kwargs.get("num_workers", 0)
#
#         # percentage of unconditional model
#         self.uncond_prob = kwargs["uncond_prob"]
#
#
#     def _train_dataloader(self):
#         return data.DataLoader(dataset=self.train_data, batch_size=self.batch_size, shuffle=True, collate_fn=self._colate, num_workers=self.num_workers)
#
#     def _val_dataloader(self):
#         return data.DataLoader(dataset=self.valid_data, batch_size=1, shuffle=False, collate_fn=self._colate, num_workers=self.num_workers)
#
#     def _test_dataloader(self):
#         return data.DataLoader(dataset=self.test_data, batch_size=1, shuffle=False, num_workers=self.num_workers)
#
#
#     def _colate(self, batch):
#         """
#         batch : is a list of tuples. every row in the batch is seperate sample and every row is
#         """
#
#         new_batch = []
#
#         if len(batch) > 1:
#             raise ("batch len should euqual to 1")
#
#         for full_sample in batch:
#             audio, vertice, template, one_hot, file_name = full_sample
#             seq_len = vertice.shape[0]
#             p = random.random()
#             if p <= self.uncond_prob: # crop them and set audio to zero frames
#                 # choose a random starting point for the seqs
#                 possible_idxs = np.arange(0, seq_len - self.window)
#                 np.random.shuffle(possible_idxs)
#                 num_of_samples = min(len(possible_idxs), self.num_patches_per_seq)
#
#                 for id in range(num_of_samples):
#                     start_idx = possible_idxs[id]
#                     end_idx = start_idx + self.window
#                     sampled_vertice = vertice[start_idx:end_idx]
#
#                     # sample the audio accordingly
#                     audio_start = int((start_idx / self.fps) * self.audio_sample_rate)
#                     audio_end = int((end_idx / self.fps) * self.audio_sample_rate)
#                     sampled_audio = audio[audio_start:audio_end]
#                     new_file_name = file_name.replace(".wav", "_start%03d.wav"%start_idx)
#                     new_batch.append((sampled_audio, sampled_vertice, template, one_hot, new_file_name))
#             else:
#                 seq_len = (seq_len // 8) * 8  # to enable compression
#                 sampled_vertice = vertice[:seq_len]
#                 audio_end = int((seq_len / self.fps) * self.audio_sample_rate)
#                 sampled_audio = audio[:audio_end]
#                 new_batch.append((sampled_audio, sampled_vertice, template, one_hot, file_name))
#
#         # print("len of the batch for debugging", len(new_batch))
#         # using the default collate to group the fields
#         return default_collate(new_batch)