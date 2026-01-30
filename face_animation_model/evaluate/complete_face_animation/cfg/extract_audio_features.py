import os
from omegaconf import OmegaConf
from face_animation_model.utils.init_helper import init_from_config
from tqdm import tqdm
import torch
import numpy as np
import glob

def get_latest_checkpoint(ckpt_dir, pre_fix="model") -> str:
    """
    Returns the latest checkpoint (by time) from the given directory, of either every validation step or best
    If there is no checkpoint in this directory, returns None
    :param ckpt_dir: directory of checkpoint
    :param pre_fixe: type of checkpoint, either "_every" or "_best"
    :return: latest checkpoint file
    """
    # Find all the every validation checkpoints
    # print("{}/*{}.ckpt".format(ckpt_dir,pre_fix))
    list_of_files = glob.glob("{}/{}*.pt".format(ckpt_dir,pre_fix))
    # print(list_of_files)
    latest_checkpoint = None
    if list_of_files:
        latest_checkpoint = max(list_of_files, key=os.path.getctime)
    print('Best checkpoint', latest_checkpoint)
    return latest_checkpoint

def move_batch_to_device(batch, device="cpu"):
    if type(batch) is dict:
        new_batch = {}
        for k, x in batch.items():
            if torch.is_tensor(x):
                new_batch[k] = x.to(device)
            else:
                new_batch[k] = x
    else:
        new_batch = []
        for x in batch:
            if torch.is_tensor(x):
                new_batch.append(x.to(device))
            else:
                new_batch.append(x)

    return new_batch

def extract_audio_features():

    device = "cuda"
    # # data set config
    # os.environ["HOME"] = os.path.join("/home/bthambiraja", "work")
    os.environ["HOME"] = os.path.join("/work/bthambiraja")
    data_cfg = "dev/data/process_hdtf/data_cfg.yml"
    cfg = OmegaConf.load(data_cfg)
    data =  init_from_config(cfg.data_cfg)

    out_path = os.path.join(os.path.join(os.environ["HOME"], cfg.data_cfg.params.dataset_root, "wav2vec_audio_feat_3diface"))
    os.makedirs(out_path, exist_ok=True)

    # load the model and use the prepare model function
    model_path = "/home/bthambiraja/fast/projects/motion_root/submission_models/cond_012_17_00_concat_prob10_vel10"
    # model_path = "/fast/bthambiraja/projects/motion_root/submission_models/cond_012_17_00_concat_prob10_vel10"
    model_ckpt = "model000130000"
    cfg_path = os.path.join(model_path, 'args.yml')
    cfg = OmegaConf.load(cfg_path)
    model = init_from_config(cfg.motion_model)

    if model_ckpt == "best":
        best_ckpt = get_latest_checkpoint(os.path.join(model_path, "checkpoints"))
    else:
        best_ckpt = os.path.join(model_path, "checkpoints", model_ckpt + ".pt")

    print("Current best checkpoint", best_ckpt)
    model.init_from_ckpt(path=best_ckpt)
    model = model.eval()
    model = model.to(device)

    # iterate through the dataloader and extract the audio features
    for batch in tqdm(data._test_dataloader(), desc="iterating through the batch"):

        batch = move_batch_to_device(batch, device)
        if type(batch) is dict:
            audio = batch["audio"]
            motion = batch["vertice"]
            template = batch["template"]  # b X one_hot_dim
            one_hot = batch["one_hot"]
            filename = batch["file_name"]
            gt_pose = batch["pose"]
        else:
            audio, motion, template, one_hot, filename, gt_pose = batch
        
        seq_len = gt_pose.shape[1] 
        audio_feat = model.encode_audio(audio, seq_len)
        print(filename)
        print(motion.shape)
        print(gt_pose.shape)
        print(audio_feat.shape)


        ## dump the features as npy
        out_file = os.path.join(out_path, filename[0].replace(".wav", ".npy"))
        np.save(out_file, audio_feat[0].detach().cpu().numpy())
        print(out_file)


if  __name__ == "__main__":
    extract_audio_features()