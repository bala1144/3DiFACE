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
from scipy.spatial.transform import Rotation
import scipy

class Dataset_v2(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data,
                 subjects_dict,
                 data_type="train",
                 **kawargs):
        self.data = data # list
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type
        self.one_hot_labels = np.eye(kawargs["num_iden_cls"])
        self.window = kawargs["window"]
        self.patches_per_seq = kawargs["num_patches_per_seq"]
        self.fps = kawargs["fps"]
        self.audio_sample_rate = kawargs["audio_sample_rate"]

        self.increments = np.repeat(np.arange(self.window)[None, :], self.patches_per_seq, axis=0).reshape(self.patches_per_seq, self.window)
        audio_feat_window = int(self.audio_sample_rate * (self.window / self.fps))
        self.a_increments = np.repeat(np.arange(audio_feat_window)[None, :], self.patches_per_seq, axis=0).reshape(self.patches_per_seq,
                                                                                                        audio_feat_window)
    def set_device(self, device_id):
        self.device_id = device_id

    def get_data(self, index):
        return self.data[index]

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        local_data = self.get_data(index)
        file_name = local_data["name"]
        audio = torch.FloatTensor(local_data["audio"])
        vertice = torch.FloatTensor(local_data["vertice"])
        template = torch.FloatTensor(local_data["template"])
        one_hot = torch.FloatTensor(local_data["one_hot"])
        pose = torch.FloatTensor(local_data["pose"])

        return (audio,
                vertice,
                template,
                one_hot,
                file_name,
                pose)

    def __len__(self):
        return self.len


### hdtf metrics
mean = torch.tensor([-0.05855013, -0.0028288, -0.01964026]).float().reshape(1,3)
std = torch.tensor([0.11084834, 0.10225139, 0.08779355]).float().reshape(1,3)
data_min = torch.tensor([-0.5517633,  -0.47712454, -0.44358182]).float().reshape(1,3)
data_max = torch.tensor([0.39624417, 0.5162673,  0.3789773]).float().reshape(1,3)

def read_data(
        dataset,
        dataset_root,
        pkl_path,
        wav_path,
        vertices_path,
        template_file,
        sequence_for_training=None,
        sequence_for_validation=None,
        sequence_for_testing=None,
        normalize_pose= False,
        normalize_dim="xyz",
        to_6d_rotation=False,
        max_seq_len=900,
        smooth_sigma=2,
        standardize_data=False,
        pretrained_wav2vec_audio_feat_path=None,
        **kwargs
):

    print("Loading data...")
    data = defaultdict(dict)
    train_data = []
    valid_data = []
    test_data = []

    audio_path = os.path.join(os.getenv("HOME"), dataset_root, wav_path)
    audio_feat_path = os.path.join(os.getenv("HOME"), dataset_root, 
                                   wav_path.replace("wav", "audio_feat_path"))
    os.makedirs(audio_feat_path, exist_ok=True)

    if pretrained_wav2vec_audio_feat_path is not None:
         pretrained_wav2vec_audio_feat_path = os.path.join(os.getenv("HOME"), dataset_root, pretrained_wav2vec_audio_feat_path)

    dict_path = os.path.join(os.getenv("HOME"), dataset_root, pkl_path)
    wav2vec_path = os.path.join(os.getenv("HOME"),
                                "projects/dataset/voca_face_former",
                                "wav2vec2-base-960h")

    if not os.path.isdir(wav2vec_path):
        print("Using global processor to process the model")
        wav2vec_path = "facebook/wav2vec2-base-960h"
    else:
        print("using local processor")
    processor = Wav2Vec2Processor.from_pretrained(wav2vec_path)

    template_file = os.path.join(os.getenv("HOME"), dataset_root, template_file)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin,encoding='latin1')

    list_dir = sorted([x.split(".")[0] for x in os.listdir(dict_path)])
    subjects_dict = {}
    subjects_dict["train"] = [list_dir[i] for i in range(sequence_for_training[0], sequence_for_training[1])]
    subjects_dict["val"] = [list_dir[i] for i in range(sequence_for_validation[0], sequence_for_validation[1])]
    subjects_dict["test"] = [list_dir[i]for i in range(sequence_for_testing[0], sequence_for_testing[1])]

    all_subjects = set(subjects_dict["train"] + subjects_dict["val"] + subjects_dict["test"])
    print("Total subjects", len(all_subjects))
    for subj in tqdm(all_subjects, desc="loading subjects from disk"):

        f = f"{subj}.wav"
        wav_path = os.path.join(audio_path,  f)
        processed_audio_feat_file = os.path.join(
            audio_feat_path,
            f.replace(".wav", ".npy")
        )

        if pretrained_wav2vec_audio_feat_path is not None: ## load pretrained feat
            processed_audio_feat_file = os.path.join(
            pretrained_wav2vec_audio_feat_path,
            f.replace(".wav", ".npy")
            )
            input_values = np.load(processed_audio_feat_file)
        elif os.path.isfile(processed_audio_feat_file):
            input_values = np.load(processed_audio_feat_file).reshape(-1)
        else:
            speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
            input_values = np.squeeze(processor(speech_array,sampling_rate=16000).input_values).reshape(-1)
            print(f"saving processed  {f} to {processed_audio_feat_file}")
            np.save(processed_audio_feat_file, input_values)

        key = subj
        data[key]["audio"] = input_values
        temp = templates[subj]
        data[key]["name"] = f
        data[key]["template"] = temp.reshape((-1))

        dict_pkl = os.path.join(dict_path, f.replace("wav", "pkl"))
        with open(dict_pkl, 'rb') as fin:
            sub_dict = pickle.load(fin,encoding='latin1')
        data[key]["vertice"] = sub_dict["vertice"].reshape(-1, 15069)[:max_seq_len]  #due to the memory limit
        
        pose = sub_dict["global_pose"][:max_seq_len]
        if standardize_data:
            assert to_6d_rotation is False
            pose = (pose - mean) / std 

        data[key]["pose"] = pose_processing(pose, normalize_pose,
                                            normalize_dim, to_6d_rotation, smooth_sigma)

    if kwargs.get("train_subjects_all", None) is not None:
        subjects_dict["train_subjects_all"] = [i for i in kwargs.get("train_subjects_all", None) .split(" ")]
    else:
        subjects_dict["train_subjects_all"] = subjects_dict["train"]
        print("\nSetting the training subjecet as training subjects all in the read data file")

    for k, v in data.items():
        subject_id = k
        if subject_id in subjects_dict["train"]:
            train_data.append(v)
        if subject_id in subjects_dict["val"]:
            valid_data.append(v)
        if subject_id in subjects_dict["test"]:
            test_data.append(v)

    print("Datset distribution")
    print("Total subjects", len(all_subjects))
    print(len(train_data), len(valid_data), len(test_data))
    return train_data, valid_data, test_data, subjects_dict

from torch.utils.data._utils.collate import default_collate
class DataModuleFromConfig_windows():
    def __init__(self, **kwargs):
        super().__init__()
        ### set up the basic
        self.batch_size = kwargs.get("batch_size", 1)
        self.window = kwargs.get("window", 30)
        self.fps = kwargs.get("fps", 30)
        self.audio_sample_rate = kwargs.get("audio_sample_rate", 16000)
        self.num_workers = kwargs.get("num_workers", 0)
        self.num_patches_per_seq = kwargs.get("num_patches_per_seq", 5)
        self.max_seq_len = kwargs.get("max_seq_len", 900)
        self.precompute_window_samples = kwargs.get("precompute_window_samples", False)
        self.sample_type = kwargs.get("sample_type", "random_crop")
        self.standardize_data = kwargs.get("standardize_data", "False")
        self.pretrained_wav2vec_audio_feat_path = kwargs.get("pretrained_wav2vec_audio_feat_path", None)

        if self.standardize_data:
            self.mean = mean
            self.std = std

        if self.window > self.max_seq_len:
            self.window = self.max_seq_len
            print("********* setting the window to the max seq lem")

        # load the data
        train_data, valid_data, test_data, subjects_dict = read_data(**kwargs)
        self.num_iden_cls = kwargs["num_iden_cls"]
        self.use_identity = kwargs["use_identity"]

        self.subjects_dict = subjects_dict
        train_data = self.process_seq(train_data, self.window, "train", self.precompute_window_samples) 
        valid_data = self.process_seq(valid_data, self.window, "val", self.precompute_window_samples)
        test_data = self.process_seq(test_data, self.window, "test", False)

        dataset_type = Dataset_v2

        self.train_data = dataset_type(train_data, subjects_dict, "train", **kwargs)
        self.valid_data = dataset_type(valid_data, subjects_dict, "val",  **kwargs)
        self.test_data = dataset_type(test_data, subjects_dict, "test",  **kwargs)

        # need for the pytorhch lightning modeule to work
        self.dataset_configs = {}
        self.dataset_configs["train"] = self.train_data
        self.train_dataloader = self._train_dataloader
        self.dataset_configs["validation"] = self.valid_data
        self.val_dataloader = self._val_dataloader
        self.dataset_configs["test"] = self.test_data
        self.test_dataloader = self._test_dataloader
        self.device_id = None

        #### hdtf metrics
        # mean = torch.tensor([-0.05855013, -0.0028288, -0.01964026]).float().reshape(1,3)
        # std = torch.tensor([0.11084834, 0.10225139, 0.08779355]).float().reshape(1,3)
        # min = torch.tensor([-0.5517633,  -0.47712454, -0.44358182]).float().reshape(1,3)
        # max = torch.tensor([0.39624417, 0.5162673,  0.3789773]).float().reshape(1,3)
    
    def process_seq(self, data:list, window:int, dataset:str, precompute):

        one_hot_labels = np.eye(self.num_iden_cls)

        if dataset == "train":
            possible_train_subjects = sorted(self.subjects_dict["train_subjects_all"])
            print("************** All train subjects ************** \n", "\n".join(sorted(possible_train_subjects)))
            assert len(possible_train_subjects) <= self.num_iden_cls
 
        print(f"\nProcessing {dataset} precompute {precompute}")

        new_data = []
        for seq_dict in data:
            seq_len = seq_dict["vertice"].shape[0]
            seq_len =  seq_len // 32 * 32
            
            file_name = seq_dict["name"]
            vertice = seq_dict["vertice"]
            audio = seq_dict["audio"]
            template = seq_dict["template"]
            pose = seq_dict["pose"]

            if self.use_identity:
                # compute the subject
                subject_id = file_name.split(".wav")[0]
                if subject_id in self.subjects_dict["train_subjects_all"]:
                    idx = sorted(self.subjects_dict["train_subjects_all"]).index(subject_id)
                    one_hot = one_hot_labels[idx]
                    new_subject_id = f"sid{idx+1}"
                    print("new_subject_id", subject_id, new_subject_id)
                else:
                    one_hot = one_hot_labels[0]
                    new_subject_id = f"sid{dataset}"
                file_name = file_name.replace(".wav", f"_{new_subject_id}.wav")
            else:
                one_hot = one_hot_labels[0]

            if window > seq_len and self.batch_size > 1:
                ### if window is greater the seq len, we cannot process it
                ### also it will causes issues during batching samples together
                print(f'Skipping {seq_dict["name"]}, win {window}, seq len {seq_len}, window bigger than seqlen')
                continue

            ### rewriting it amke it simple
            elif seq_len > window and precompute: ### precompute the crops
                
                if self.sample_type == "random_crop":
                    possible_idxs = np.arange(0, seq_len - self.window)
                    np.random.shuffle(possible_idxs)
                    num_of_samples = min(len(possible_idxs), self.num_patches_per_seq)
                    start_idxs = possible_idxs[:num_of_samples]
                elif self.sample_type == "uniform_sample_with_overlap": ### warning don't use this, this will make it super big
                    unique_parts = len(np.arange(0, seq_len, self.window)) # number of unique parts
                    start_idxs = np.int64(np.linspace(0, seq_len-window, 4 * unique_parts))
                elif self.sample_type == "uniform_sample_no_overlap":
                    start_idxs = np.arange(0, seq_len, self.window)

                audio = torch.from_numpy(audio)
                audio_feat_window = int(self.audio_sample_rate * (window / self.fps))
                for idx in start_idxs:

                    start = idx
                    end = idx + window
                    i_vertice = vertice[start:end, :]
                    i_pose = pose[start:end, :]
                    i_file = file_name.replace(".wav", "_start%03d.wav" % idx)

                    if self.pretrained_wav2vec_audio_feat_path is None: ## handle load audio features
                        audio_start = int(idx / self.fps * self.audio_sample_rate)
                        audio_end = audio_start + audio_feat_window
                        i_audio = audio[audio_start:audio_end]
                    else:
                        i_audio = audio[start:end, :]

                    new_seq = {
                        "audio": i_audio,
                        "name": i_file,
                        "vertice": i_vertice,
                        "template": template,
                        "one_hot": one_hot,
                        "pose": i_pose,
                        }
                    new_data.append(new_seq)

            else: ## crop th sequence lengthe

                sampled_vertice = vertice[:seq_len]
                sampled_pose = pose[:seq_len]

                if self.pretrained_wav2vec_audio_feat_path is None: ## handle load audio features
                    audio_end = int((seq_len / self.fps) * self.audio_sample_rate)
                    sampled_audio = audio[:audio_end]
                else:
                    sampled_audio = audio[:seq_len]

                new_seq = {
                    "audio": sampled_audio,
                    "name": file_name,
                    "vertice": sampled_vertice,
                    "template": template,
                    "one_hot": one_hot,
                    "pose": sampled_pose,
                }
                new_data.append(new_seq)

        print(f"Precomputing windows {dataset}, org data len {len(data)}, new data len {len(new_data)}")
        assert len(new_data) >= len(data) 

        return new_data
    
    def dump_subject_id(self):
        pass

    def set_device(self, device_id):
        self.device_id = device_id
        self.mean = self.mean.to(device_id)
        self.std = self.std.to(device_id)

    def norm_to_metric_space(self, x, device="cpu"):
        return (x * self.std) + self.mean  

    def metric_space_to_norm(self, x):
        return (x - self.mean) / self.std  


    def _train_dataloader(self, shuffle=True, collate=True):
    
        collate_fn = self._colate if collate else None
        return data.DataLoader(dataset=self.train_data, batch_size=self.batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=self.num_workers)

    def _val_dataloader(self):
        return data.DataLoader(dataset=self.valid_data, batch_size=self.batch_size, shuffle=False, collate_fn=self._colate, num_workers=self.num_workers)

    def _test_dataloader(self):
        self.test_max_seq_len = 600
        self.test_max_seq_len =  (self.test_max_seq_len // 32) * 32
        return data.DataLoader(dataset=self.test_data, batch_size=1, shuffle=False, collate_fn=self._test_colate, num_workers=self.num_workers)

    def _colate(self, batch):
        """
        batch : is a list of tuples. every row in the batch is seperate sample and every row is
        """
        audio, vertice, template, one_hot, file_name, pose = default_collate(batch)

        if len(audio.shape) == 3 and self.pretrained_wav2vec_audio_feat_path is None:  # handle the cropped case
            bs, t, _ = audio.shape
            audio = audio.reshape(bs * t, -1)
            vertice = vertice.reshape(bs*t, self.window, -1)
            pose = pose.reshape(bs*t, self.window, -1)
            template = template.reshape(bs * t, -1)
            one_hot = one_hot.reshape(bs*t, -1)
            file_name = list(itertools.chain(*file_name))
        
        return audio, vertice, template, one_hot, file_name, pose
    
    def _test_colate(self, batch):
        """
        batch : is a list of tuples. every row in the batch is seperate sample and every row is
        """
        audio, vertice, template, one_hot, file_name, pose = default_collate(batch)
        seq_len = min(vertice.shape[1], self.test_max_seq_len)

        sampled_vertice = vertice[:, :seq_len]
        sampled_pose = pose[:, :seq_len]
        if self.pretrained_wav2vec_audio_feat_path is None: ## handle load audio features
            audio_end = int((seq_len / self.fps) * self.audio_sample_rate)
            sampled_audio = audio[:audio_end]
        else:
            sampled_audio = audio[:seq_len]

        return sampled_audio, sampled_vertice, template, one_hot, file_name, sampled_pose


from face_animation_model.utils.torch_rotation import *
def pose_processing(pose, normalize_pose, normalize_dim, to_6d_rotation, sigma=2):

    if not torch.is_tensor(pose):
        pose = torch.from_numpy(pose).float()

    if sigma > 0.00000001:
        pose = scipy.ndimage.filters.gaussian_filter1d(pose, sigma=sigma, axis=0)
        pose = torch.from_numpy(pose).float()

    if normalize_pose:
        pose_matrix = axis_angle_to_matrix(pose)
        norm_pose_pose_matrix = torch.matmul(torch.linalg.inv(pose_matrix[0]), pose_matrix[:])
        norm_pose = matrix_to_axis_angle(norm_pose_pose_matrix)
    else:
        norm_pose = pose

    if to_6d_rotation:
        norm_pose = axis_angle_to_6D(norm_pose)

    if torch.is_tensor(pose):
        return norm_pose
    else:
        return norm_pose.numpy()
