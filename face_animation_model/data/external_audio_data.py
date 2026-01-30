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
import glob

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data,
                        data_type="train",
                        number_identity_cls=8, 
                        default_init_index=2):
        self.data = data
        self.len = len(self.data)
        self.data_type = data_type
        self.one_hot_labels = np.eye(number_identity_cls)
        self.custom_init_id = default_init_index
        self.default_index=default_init_index

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        file_name = self.data[index]["name"]
        audio = self.data[index]["audio"]
        vertice = self.data[index]["vertice"]
        template = self.data[index]["template"]
        one_hot = self.one_hot_labels[self.default_index]
        return torch.FloatTensor(audio),torch.FloatTensor(vertice), torch.FloatTensor(template), torch.FloatTensor(one_hot), file_name

    def __len__(self):
        return self.len

def read_data(
        dataset,
        dataset_root,
        wav_path,
        wav2vec_path,
        template_file,
        load_all_files,
        audio_files_to_test,
        condition_for_tesing,
        **kwargs
):
    print("Loading data...")
    data = defaultdict(dict)
    test_data = []

    audio_path = wav_path
    if not os.path.isdir(wav2vec_path):
        print("Using global processor to process the model")
        wav2vec_path = "facebook/wav2vec2-base-960h"
    else:
        print("using local processor")
    processor = Wav2Vec2Processor.from_pretrained(wav2vec_path)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin,encoding='latin1')

    if load_all_files:
        all_wav_files = sorted(glob.glob(os.path.join(audio_path, "*.wav")))
    else:
        audio_files_to_test = audio_files_to_test.split(" ")
        all_wav_files = [os.path.join(audio_path, x+".wav") for x in audio_files_to_test]

    for subject_id in condition_for_tesing.split(" "):
        for wav_path in tqdm(all_wav_files):
            f = wav_path.split("/")[-1]
            if f.endswith("wav"):
                wav_path = os.path.join(audio_path,f)
            speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
            input_values = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)

            nf = int((input_values.shape[0] / 16_000) * 30)
            key = f.replace("wav", "npy")
            data[key]["audio"] = input_values

            temp = templates[subject_id]
            data[key]["name"] = wav_path
            data[key]["template"] = temp.reshape((-1))
            data[key]["vertice"] = np.tile(temp.reshape(1,-1), (nf, 1))
            test_data.append(data[key])

    return test_data

class DataModuleFromConfig():
    def __init__(self, **kwargs):
        super().__init__()
        test_data = read_data(**kwargs)
        num_iden_cls = kwargs["num_iden_cls"]
        default_init_index =  kwargs["default_init_index"]
        self.test_data = Dataset(test_data, "test", num_iden_cls, default_init_index)

        # need for the pytorhch lightning modeule to work
        self.dataset_configs = {}
        self.dataset_configs["test"] = self.test_data
        self.test_dataloader = self._test_dataloader

    def prepare_data(self):
        print("prepere data do nothin")

    def setup(self, stage=None):
        print("setup do nothin")



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


    def _train_dataloader(self):
        # return data.DataLoader(dataset=self.train_data, batch_size=self.batch_size, shuffle=True, collate_fn=self._colate, num_workers=self.num_workers)
        return

    def _val_dataloader(self):
        # return data.DataLoader(dataset=self.valid_data, batch_size=1, shuffle=False, collate_fn=self._colate, num_workers=self.num_workers)
        return

    def _test_dataloader(self):
        return data.DataLoader(dataset=self.test_data, batch_size=1, collate_fn=self._colate, shuffle=False, num_workers=self.num_workers)

    def _colate(self, batch):
        """
        batch : is a list of tuples. every row in the batch is seperate sample and every row is
        """

        new_batch = []

        for full_sample in batch:
            audio, vertice, template, one_hot, file_name = full_sample
            seq_len = vertice.shape[0]
            if self.window < seq_len and len(batch) > 1:
                # choose a random starting point for the seqs
                possible_idxs = np.arange(0, seq_len - self.window)
                np.random.shuffle(possible_idxs)

                for id in range(self.num_patches_per_seq):
                    start_idx = possible_idxs[id]
                    end_idx = start_idx + self.window
                    sampled_vertice = vertice[start_idx:end_idx]
                    # sample the audio accordingly
                    audio_start = int((start_idx / self.fps) * self.audio_sample_rate)
                    audio_end = int((end_idx / self.fps) * self.audio_sample_rate)
                    sampled_audio = audio[audio_start:audio_end]

                    # compose the new batch
                    new_batch.append((sampled_audio, sampled_vertice, template, one_hot, file_name))
            else:
                seq_len = (seq_len // 8) * 8  # to enable compression
                sampled_vertice = vertice[:seq_len]
                audio_end = int((seq_len / self.fps) * self.audio_sample_rate)
                sampled_audio = audio[:audio_end]
                new_batch.append((sampled_audio, sampled_vertice, template, one_hot, file_name))

        # using the default collate to group the fields
        return default_collate(new_batch)