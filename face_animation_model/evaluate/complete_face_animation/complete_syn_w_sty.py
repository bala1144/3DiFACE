import os, datetime, glob, importlib
from omegaconf import OmegaConf
import numpy as np
import json
import collections, functools, operator
from argparse import ArgumentParser
import torch
torch.backends.cudnn.deterministic = True
from face_animation_model.utils.fixseed import fixseed
from face_animation_model.utils.parser_util import *
from face_animation_model.utils import dist_util
from face_animation_model.train.training_loop import TrainLoop
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform

from tqdm import tqdm
import pickle
from face_animation_model.utils.fixseed import fixseed
from scipy.spatial.transform import Rotation
from glob import glob
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
title_font = {'fontname': 'DejaVu Sans', 'size': '16', 'color': 'black', 'weight': 'normal',
              'verticalalignment': 'bottom'}


from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor
import librosa
import scipy

### hdtf metrics
mean = torch.tensor([-0.05855013, -0.0028288, -0.01964026]).float().reshape(1,3)
std = torch.tensor([0.11084834, 0.10225139, 0.08779355]).float().reshape(1,3)
data_min = torch.tensor([-0.5517633,  -0.47712454, -0.44358182]).float().reshape(1,3)
data_max = torch.tensor([0.39624417, 0.5162673,  0.3789773]).float().reshape(1,3)


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def init_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

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

def process_result_dict(results_dict):
    # combine the results dicts
    loss_dicts = [batch['results'] for batch in results_dict]
    # add
    combined = dict(functools.reduce(operator.add, map(collections.Counter, loss_dicts)))
    # average the loss
    average_loss = {key: combined[key] / len(loss_dicts) for key in combined.keys()}

    return average_loss


class test_w_sty_head():

    def __init__(self,
                 **kwargs,
                ) -> None:
        super().__init__(**kwargs)

        config = {}
        if len(config) == 0:
            config["flame_model_path"] = os.path.join("./FLAMEModel",
                                                      "model/generic_model.pkl")
            config["batch_size"] = 1
            config["shape_params"] = 0
            config["expression_params"] = 0
            config["pose_params"] = 0
            config["number_worker"] = 8
            config["use_3D_translation"] = False

        from FLAMEModel.FLAME import FLAME
        self.face_model = FLAME(config)

        if os.getenv("WAV2VEC_PATH"):
            wav2vec_path = os.getenv("WAV2VEC_PATH")
        else:
            wav2vec_model = "projects/dataset/voca_face_former/wav2vec2-base-960h"
            wav2vec_path = os.path.join(os.getenv('HOME'), wav2vec_model)
        
        self.processor = Wav2Vec2Processor.from_pretrained(wav2vec_path)

        self.templates = None
        from face_animation_model.visualizer.render_head_video_from_dump import render_helper
        self.render_helper_obj = render_helper(render_type="normal_camera")

    def load_templates(self, method):

        if method == "voca":
            if os.getenv("VOCASET_PATH"):
                template_file = os.path.join(os.getenv("VOCASET_PATH"), "templates.pkl")
            else:
                template_file = os.path.join(os.getenv("HOME"), "projects/dataset/voca_face_former", "templates.pkl")
                with open(template_file, 'rb') as handle:
                    self.templates = pickle.load(handle, encoding='latin1')

        elif method == "HDTF":
            template_file = os.path.join(os.getenv("HOME"), "projects/dataset/HDTF", "templates.pkl")
            with open(template_file, 'rb') as handle:
                self.templates = pickle.load(handle, encoding='latin1')

    def read_audio_from_file(self, wav_path):
        speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
        input_values = np.squeeze(self.processor(speech_array, sampling_rate=16000).input_values)
        return input_values

    def move_batch_to_device(self, batch, device=None):
        if device is None:
            device = self.device
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

    def plot_curves(self, all_pose:list, model_names:list, out_file):

        import matplotlib.pyplot as plt
        title_font = {'fontname': 'DejaVu Sans', 'size': '16', 'color': 'black', 'weight': 'normal',
                      'verticalalignment': 'bottom'}


        ## create the graph with mean
        fig, ax = plt.subplots(figsize=(15,10), nrows=4)
        gt_pose = all_pose[0]
        if len(gt_pose.shape) == 3:
            gt_pose=gt_pose[0]
            
        x = np.arange(gt_pose.shape[0])
        for d in [3,0,1,2]:
            # plot mea
            color_cycler = iter(plt.cm.tab10.colors)
            for i, y in enumerate(all_pose):
                
                if len(y.shape) == 3:
                    y = y[0].detach().cpu()
                else:
                    y = y.detach().cpu()

                if d == 3:
                    ax[3].plot(x, torch.mean(y, -1), color=next(color_cycler), label=f"mean_{model_names[i]}")
                else:
                    ax[d].plot(x, y[:, d], color=next(color_cycler), label=f"ss_{model_names[i]}_ax{d}")
                
                ax[d].legend(loc='best')
        
        label = "Axis wise"
        plt.title(f"{label}_stat", **title_font)
        plt.xticks(np.arange(0, x.shape[0], 30))
        plt.ylabel('Angle(deg)')
        plt.tight_layout()
        fig.savefig(out_file.replace(".png", "_ax_wise.png"), dpi=300)
        plt.close(fig)
        print("storing", out_file.replace(".png", "_ax_wise.png"))

    def get_euler_from_6D(self, pose):
        b1 = pose[:, :3]
        b2 = pose[:, 3:]
        b3 = torch.cross(b1, b2, dim=-1)
        rot_matrix = torch.stack((b1, b2, b3), dim=-2).detach().cpu().numpy()
        return Rotation.from_matrix(rot_matrix).as_euler("xyz", degrees=True)

    def compute_pose_diff(self, gt, pred):
        """
        gt : T x 3
        pred : T x 3
        """
        if gt.shape[-1] == 6:
            gt_pose = self.get_euler_from_6D(gt)
            pred_pose = self.get_euler_from_6D(pred)
        else:
            gt_pose = Rotation.from_rotvec(gt.cpu().numpy()).as_euler("xyz", degrees=True)
            pred_pose = Rotation.from_rotvec(pred.cpu().numpy()).as_euler("xyz", degrees=True)
        diff_pose = gt_pose - pred_pose
        return torch.from_numpy(Rotation.from_euler("xyz", diff_pose, degrees=True).as_rotvec()).float().to(gt.device)

    def prepare_model(self, args, ext_str=None):
        
        exp_cfg=args.exp_cfg
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

        ckpt_name = self.head_motion_ckpt.split("/")[-1].split(".")[0]
        if exp_cfg.use_time_stamp:
            exp_cfg.experiment_name = now + "_" + exp_cfg.experiment_name + "_" + ckpt_name
        else:
            exp_cfg.experiment_name = exp_cfg.experiment_name + "_" + ckpt_name

        if ext_str is not None:
            exp_cfg.experiment_name += f"_{ext_str}"

        if exp_cfg.method == "both" or exp_cfg.method == "head":
            args.logdir = args.head_motion_model.model_path
        else:
            args.logdir = args.face_motion_model.model_path
        

        out_dir = os.path.join(args.logdir, exp_cfg.experiment_name)
        os.makedirs(out_dir, exist_ok=True)
        cfg_path = os.path.join(out_dir, 'args.yml')

        args.exp_cfg = exp_cfg
        OmegaConf.save(vars(args), cfg_path)

        print("Config path",cfg_path )

        return out_dir

    def create_audio_feature_extraction_model(self, args):
        model_path = "./pretrained_models/cond_012_17_00_concat_prob10_vel10"
        model_ckpt = "model000130000"
        cfg_path = os.path.join(model_path, 'args.yml')
        cfg = OmegaConf.load(cfg_path)
        model = init_from_config(cfg.motion_model)
        best_ckpt = os.path.join(model_path, "checkpoints", model_ckpt + ".pt")
        print("Audio feature extractor", best_ckpt)
        model.init_from_ckpt(path=best_ckpt)
        model = model.eval()
        model = model.to(args.device)
        self.audio_feat_extractor = model

    def create_expresssion_model(self, args):

        # model_path = "./checkpoints/cond_012_17_00_concat_prob10_vel10"
        model_path = args.face_motion_model.model_path
        model_ckpt = args.face_motion_model.model_ckpt
        cfg_path = os.path.join(model_path, 'args.yml')
        cfg = OmegaConf.load(cfg_path)
        cfg.motion_model.params.init_from_ckpt=None
        model = init_from_config(cfg.motion_model)
        if model_ckpt == "best":
            best_ckpt = get_latest_checkpoint(os.path.join(model_path, "checkpoints"))
        else:
            best_ckpt = os.path.join(model_path, "checkpoints", model_ckpt + ".pt")

        print("Current best checkpoint", best_ckpt)
        model.init_from_ckpt(path=best_ckpt)
        model = model.eval()
        model = model.to(args.device)

        guidance_model = model
        if args.face_motion_model.guidance_sampling:
            from face_animation_model.model.cfg_sampler import ClassifierFreeSampleModel
            guidance_model = ClassifierFreeSampleModel(model, args.face_motion_model.guidance_scale)

        self.face_motion_model = guidance_model
        self.create_audio_feature_extraction_model(args)

        ### create the diffusion model for sampling
        print("creating the expression-diffusion model")
        self.face_motion_diffusion = init_from_config(cfg.diffusion)
        self.face_motion_sampling_fn = self.face_motion_diffusion.p_sample_loop

    def create_head_motion_model(self, args):

        print("creating model and diffusion...")
        cfg_path = os.path.join(args.head_motion_model.model_path, 'args.yml')
        cfg = OmegaConf.load(cfg_path)
        model = init_from_config(cfg.motion_model)

        if args.head_motion_model.model_ckpt == "best":
            best_ckpt = get_latest_checkpoint(os.path.join(args.head_motion_model.model_path, "checkpoints"))
        else:
            best_ckpt = os.path.join(args.head_motion_model.model_path, "checkpoints", args.head_motion_model.model_ckpt+ ".pt")

        print("Current best checkpoint", best_ckpt)
        self.head_motion_ckpt = best_ckpt
        model.init_from_ckpt(path=best_ckpt)
        model = model.eval()
        model = model.to(args.device)
        self.head_motion_model = model

        self.face_model = self.face_model.to(args.device)

        print("creating the head motion diffusion model")
        self.head_diffusion = init_from_config(cfg.diffusion)
        self.head_diffusion_sampling_fn =  self.head_diffusion.p_sample_loop

    def set_device(self, device_id):
        self.device_id = device_id
        self.mean = mean.to(device_id)
        self.std = std.to(device_id)

    def norm_to_metric_space(self, x, device="cpu"):
        return (x * self.std) + self.mean  
    
    def metric_space_to_norm(self, x, device="cpu"):
        return (x - self.mean) / self.std  
    
    def prepare_batch(self, args, audio_file):

        ### set the condition and the model
        identity_for_testing = args.face_motion_model.identity_for_testing
        condition_for_tesing = args.face_motion_model.condition_for_tesing
        exp_cfg = args.exp_cfg
        
        if "HDTF" in audio_file and not args.face_motion_model.use_voca_template_for_vis:
            if self.templates is None:
                self.load_templates("HDTF")
            subj = audio_file.split("/")[-1].split(".")[0]
            template= self.templates[subj]
            template = template.reshape(1, -1).float()
            index = args.face_motion_model.condition_index

        
        else: ### for the vocaset
            
            if self.templates is None:
                self.load_templates("voca")
            template = self.templates[identity_for_testing]
            template = torch.from_numpy(template.reshape(1, -1)).float()
            # index = self.face_motion_model.train_subjects.index(condition_for_tesing)
            index = args.face_motion_model.condition_index

        one_hot = torch.zeros((1, 8))
        one_hot[0, index] = 1

        # process the audio for the audio features similar motion synthesis
        processed_audio = self.read_audio_from_file(audio_file)
        sampled_processed_audio = torch.from_numpy(processed_audio).view(1, -1)

        nf = (sampled_processed_audio.shape[1] // 16_000) * 30
        nf = (nf // 32) * 32 # this is done to accomate the convolutional compression
        motion =  torch.zeros(1, nf, 15069).float()
        af = int(nf / 30 * 16_000)
        sampled_processed_audio = sampled_processed_audio[:, :af]

        sampled_processed_audio = sampled_processed_audio.to(args.device)
        motion = motion.to(args.device)
        template = template.to(args.device)
        one_hot = one_hot.to(args.device)
        new_batch = (sampled_processed_audio,
                    motion,
                    template,
                    one_hot,
                    audio_file
                    )

        return new_batch
    
    def get_audio_feature(self, args, batch):
        new_batch = []
        for x in batch:
            if torch.is_tensor(x):
                new_batch.append(x.to(args.device))
            else:
                new_batch.append(x)
        batch = new_batch
        if len(batch) == 5:
            audio, motion, template, one_hot_input, file_name = batch
        else:
            audio, motion, template, one_hot_input, file_name, pred_pose = batch

        shape = motion.shape
        nf = motion.shape[1]
        audio_feat = self.audio_feat_extractor.encode_audio(audio, nf)
        return audio_feat
  
    def sample_exprs(self, args, batch, y=None):

        new_batch = []
        for x in batch:
            if torch.is_tensor(x):
                new_batch.append(x.to(args.device))
            else:
                new_batch.append(x)
        batch = new_batch
        
        if len(batch) == 5:
            audio, motion, template, one_hot_input, file_name = batch
        else:
            audio, motion, template, one_hot_input, file_name, pred_pose = batch

        shape = motion.shape
        nf = motion.shape[1]
        model_kwargs={"batch": batch}
        if y is not None:
            model_kwargs['y'] = y
        prediction = self.face_motion_sampling_fn(self.face_motion_model,
                            shape,
                            progress=True,
                            dump_steps=None,
                            model_kwargs=model_kwargs)

        expression = prediction[-1].reshape(-1, 5023, 3).detach() # exprs
        audio_feat = self.audio_feat_extractor.encode_audio(audio, nf)
            
        assert expression.shape[0] == audio_feat.shape[1]
        return expression, audio_feat

    def sample_head_motion(self, batch, audio_feat):
        
        ## batch
        audio, motion, template, one_hot, filename = batch
        gt_pose = torch.zeros(1, motion.shape[1], 3).float().to(args.device)


        new_batch_with_condition = (audio_feat, torch.zeros_like(gt_pose), template, one_hot, filename)
        shape = gt_pose.shape
        model_kwargs = {"batch": new_batch_with_condition}
        pose_pred = self.head_diffusion_sampling_fn(self.head_motion_model,
                                shape,
                                progress=True,
                                model_kwargs=model_kwargs)

        return pose_pred
    
    def run_external_audio(self, args):
        
        self.set_device(args.device)
        self.create_expresssion_model(args)
        self.create_head_motion_model(args)

        exp_cfg = args.exp_cfg
        out_dir = self.prepare_model(args)
        out_dir_npy = os.path.join(out_dir, "npy")
        out_dir_video = os.path.join(out_dir_npy+"_video")
        os.makedirs(out_dir_npy, exist_ok=True)
        os.makedirs(out_dir_video, exist_ok=True)
        
        num_of_samples = exp_cfg.num_of_samples
        outprocess_fn = self.norm_to_metric_space

        if ".wav" in args.audio_file: # run on single_file
            to_run = [args.audio_file]
        else:
            to_run = glob(os.path.join(args.audio_file, "*.wav"))

        print("Number of files to run", len(to_run))
        for audio_file in to_run:
            print("\nRunning ", audio_file)
            new_batch = self.prepare_batch(args, audio_file)
            exprs, audio_feat = self.sample_exprs(args, new_batch)
            audio, _, template, one_hot, filename = new_batch
            
            seq_name = filename.split("/")[-1].split(".")[0]

            all_vids = []
            for i in range(num_of_samples):

                pose_pred = self.sample_head_motion(new_batch, audio_feat)
                pose_pred = outprocess_fn(pose_pred)

                ss_seq_name_w_condition = seq_name + "_ss%02d" % i
                if exp_cfg.render:
                ## visualize the results
                    curr_pred = self.face_model.apply_neck_rotation(exprs, pose_pred[0]).detach()
                    out_file, pred_rendered_images = self.render_helper_obj.visualize_meshes(out_dir_video,
                                                                                            ss_seq_name_w_condition,
                                                                                            audio_file,
                                                                                            curr_pred,
                                                                                            "ss%02d" % i)

                    all_vids.append(out_file)

                out_dict = {
                    "vertice" : exprs.detach().cpu().numpy(),
                    "pose" : pose_pred[0].detach().cpu().numpy(),
                }
                ## dump the metric with head motion
                curr_out_file = os.path.join(out_dir_npy, ss_seq_name_w_condition+".npy")
                np.save(curr_out_file, out_dict)
                print("Dumping out file", curr_out_file)

    def get_edit_batch(self, args, pred_file):

        identity_for_testing = args.face_motion_model.identity_for_testing
        condition_for_tesing = args.face_motion_model.condition_for_tesing
        
        nf = args.exp_cfg.num_frames_per_sample
        result_dict = np.load(pred_file, allow_pickle=True).item() # T x 3

        pose = result_dict["pose"][:nf]# T x 3
        pred_pose = torch.from_numpy(pose).float().reshape(1, -1, 3) # T x 3

        motion = torch.from_numpy(result_dict["vertice"][:nf]).reshape(1, -1, 15069)

        seq_name = pred_file.split("/")[-1].split(".")[0]
        seq_name_without_ss = seq_name.split("_ss")[0]

        one_hot = torch.zeros((1, 8))
        one_hot[0, 2] = 1

        # audio_file = pred_file.split("/")[-1].replace(".npy", ".wav")
        if os.path.exists(args.audio_file):
            audio_file = args.audio_file
        elif "RD_Rad" in  seq_name_without_ss:
            audio_file = os.path.join("/home/bthambiraja/work/projects", "dataset/HDTF/test_wav_20sec", seq_name_without_ss+".wav")
        else:
            audio_file = os.path.join("/home/bthambiraja/work/projects", "dataset/external_audio_test/head_motion_test", "wav", seq_name_without_ss+".wav")

        processed_audio = self.read_audio_from_file(audio_file)
        sampled_processed_audio = torch.from_numpy(processed_audio).view(1, -1)

        nf = (nf // 32) * 32 # this is done to accomate the convolutional compression
        af = int(nf / 30 * 16_000)
        sampled_processed_audio = sampled_processed_audio[:, :af]

        sampled_processed_audio = sampled_processed_audio.to(args.device)
        # motion = motion.to(args.device).view(1, nf, -1)
        if self.templates is None:
            self.load_templates("voca")
        template = self.templates[identity_for_testing]
        template = torch.from_numpy(template.reshape(1, -1)).float()
    
        template = template.to(args.device)

        one_hot = one_hot.to(args.device)
        pred_pose = pred_pose.to(args.device)
        motion = motion.to(args.device)

        new_batch = (sampled_processed_audio,
                    motion,
                    template,
                    one_hot,
                    audio_file,
                    pred_pose
                    )
        
        return new_batch
    
    def get_edit_mask(self, exp_cfg, model_kwargs, gt_pose, gt_seq_name):
        
        if exp_cfg.edit_mode == "in_between":
                # in_between_frames = [(90, 220)]
                in_between_frames = [(90, 250)]
                model_kwargs['y']['inpainted_motion'] = gt_pose  # chnage
                model_kwargs['y']['inpainting_mask'] = torch.ones_like(gt_pose,
                                                                        dtype=torch.bool,
                                                                        device=gt_pose.device)  # True means use gt motion
                for (start_idx, end_idx) in in_between_frames:
                    print("start/end", start_idx, end_idx)
                    model_kwargs['y']['inpainting_mask'][:, start_idx: end_idx, :] = False  # do inpainting in those frames
        
                    # seq_name = gt_seq_name + f"{start_idx}_{end_idx}"
                    seq_name = gt_seq_name + "%03d_%03d"%(start_idx, end_idx)
                    inpainted_frames = list(range(start_idx, end_idx))

        elif exp_cfg.edit_mode == "keyframe_swap":
            inpainted_motion = torch.zeros_like(gt_pose,
                                                device=gt_pose.device)
            
            inpainted_mask = torch.zeros_like(gt_pose, dtype=torch.bool,
                                                                    device=gt_pose.device)  # False means use inpaint motion
            
            key_frames = []
            for trow in exp_cfg.key_frames_dict:
                from_, to_ = tuple(map(int, trow.split(",")))
                inpainted_mask[:, to_, :] = True  # use GT in those frames
                inpainted_motion[:, to_, :] = gt_pose[:, from_, :]  # copy the motion
                key_frames.append(to_)

            model_kwargs['y']['inpainted_motion'] = inpainted_motion
            model_kwargs['y']['inpainting_mask'] = inpainted_mask  # True means use gt motion
            exp_cfg.key_frames = key_frames  # used for visulizatoin
            seq_name = gt_seq_name + f"_swap_kf_len_{len(exp_cfg.key_frames)}"
        
        else:
            model_kwargs['y']['inpainted_motion'] = gt_pose
            model_kwargs['y']['inpainting_mask'] = torch.zeros_like(gt_pose,
                                                                    dtype=torch.bool,
                                                                    device=gt_pose.device)  # False means use inpaint motion
            
            exp_cfg.key_frames = []
            # exp_cfg.key_frames.extend(list(range(0,gt_pose.shape[1],100)))
            exp_cfg.key_frames.extend(list(range(0,gt_pose.shape[1],60)))
            seq_name = gt_seq_name + f"_kf_len_{len(exp_cfg.key_frames)}"
            edited_pose = torch.zeros_like(gt_pose)
            for idx in exp_cfg.key_frames:
                model_kwargs['y']['inpainting_mask'][ :, idx, :] = True # use as gt frames
                edited_pose[ :, idx, :] = gt_pose[:, idx, :].clone()
        
        return model_kwargs, exp_cfg, seq_name

    def run_edit_external_audio(self, args):

        print("Running external model")
        
        self.set_device(args.device)
        self.create_audio_feature_extraction_model(args)
        self.create_expresssion_model(args)
        self.create_head_motion_model(args)


        exp_cfg = args.exp_cfg
        out_dir = self.prepare_model(args, "editing_test")
        out_dir_npy = os.path.join(out_dir, "npy")
        out_dir_video = os.path.join(out_dir_npy+"_video")
        os.makedirs(out_dir_npy, exist_ok=True)
        os.makedirs(out_dir_video, exist_ok=True)
        
        num_of_samples = exp_cfg.num_of_samples
        inprocess_fn = self.metric_space_to_norm
        outprocess_fn = self.norm_to_metric_space

        if ".npy" in args.edit_file: # run on single_file
            to_run = [args.edit_file]
        else:
            to_run = glob(os.path.join(args.edit_file, "*_ss02.npy"))

        for edit_file in to_run:
            new_batch = self.get_edit_batch(args, edit_file)

            if "face" not in exp_cfg.edit_part:
                audio, exprs, template, one_hot, filename, in_head_pose = new_batch
                gt_exprs = exprs
                seq_name = filename.split("/")[-1].split(".")[0]
                audio_file = filename
                gt_seq_name = seq_name
                exprs = exprs[0].reshape(-1, 5023, 3).detach()
            else:
                audio, gt_exprs, template, one_hot, filename, in_head_pose = new_batch
                face_batch = (audio, gt_exprs, template, one_hot, filename)
                shape = gt_exprs.shape
                model_kwargs = {"batch": face_batch}
                model_kwargs['y'] = {}
                seq_name = filename.split("/")[-1].split(".")[0]
                audio_file = filename
                gt_seq_name =  seq_name
                model_kwargs, tmp_exp_cfg, tmp_seq_name, = self.get_edit_mask(exp_cfg, model_kwargs, gt_exprs-template.unsqueeze(1), gt_seq_name)
                exprs, _ = self.sample_exprs(args, face_batch, model_kwargs['y'])
    
            audio_feat = self.get_audio_feature(args, new_batch)

            out_dict = {
                "vertice" : gt_exprs.detach().cpu().numpy(),
                "pose" : in_head_pose[0].detach().cpu().numpy(),
            }

            if exp_cfg.render:
            ## visualize the results
                vis_gt_exprs = gt_exprs[0].reshape(-1, 5023, 3).detach()
                curr_pred = self.face_model.apply_neck_rotation(vis_gt_exprs, in_head_pose[0]).detach()
                gt_file, pred_rendered_images = self.render_helper_obj.visualize_meshes(out_dir_video,
                                                                                        "gt_"+gt_seq_name,
                                                                                        audio_file,
                                                                                        curr_pred, "gt")



            ## dump the metric with head motion
            curr_out_file = os.path.join(out_dir_npy, "gt_"+seq_name+".npy")
            np.save(curr_out_file, out_dict)
            print("Dumping out file", curr_out_file)

            gt_pose_normalized = inprocess_fn(in_head_pose) # metric to norm
            new_batch_with_condition = (audio_feat, gt_pose_normalized, template, one_hot, filename)
            shape = gt_pose_normalized.shape
            model_kwargs = {"batch": new_batch_with_condition}
            model_kwargs['y'] = {}
            gt_seq_name =  seq_name
            model_kwargs, exp_cfg, seq_name, = self.get_edit_mask(exp_cfg, model_kwargs, gt_pose_normalized, gt_seq_name)

            for i in range(exp_cfg.num_of_samples):
                pose_pred_normed_space = self.head_diffusion_sampling_fn(self.head_motion_model,
                                        shape,
                                        progress=True,
                                        model_kwargs=model_kwargs)
                
                ## norm space
                ss_seq_name_w_condition = seq_name + "_ss%02d" % i
                if exp_cfg.graph_vis:

                    out_dir_graph = os.path.join(out_dir_npy+"_graph")
                    os.makedirs(out_dir_graph, exist_ok=True)
                    
                    ## create the output folder
                    T = gt_pose_normalized.shape[1]
                    out_vis_file = os.path.join(out_dir_graph, ss_seq_name_w_condition+".png")
                    time = np.arange(0, T, 1)

                    color_cycler = iter(plt.cm.tab10.colors)                
                    fig, ax = plt.subplots(figsize=(15,5), nrows=1)
                    ax.plot(time, np.mean(gt_pose_normalized[0].cpu().numpy(), axis=-1), color=next(color_cycler), label="Input Signal")
                    ax.plot(time, np.mean(pose_pred_normed_space[0].cpu().numpy(), axis=-1), color=next(color_cycler), label="Inpainted")
                    
                    if exp_cfg.edit_mode == "keyframe":
                        x = exp_cfg.key_frames
                        y = np.mean(gt_pose_normalized[0].cpu().numpy(), axis=-1)
                        y = [y[j] for j in x] 
                        ax.scatter(x, y, marker='x', c="magenta")

                    elif exp_cfg.edit_mode == "keyframe_swap":
                        src_curve = np.mean(gt_pose_normalized[0].cpu().numpy(), axis=-1)
                        for trow in exp_cfg.key_frames_dict:
                            from_, to_ = tuple(map(int, trow.split(",")))
                            ax.plot(from_, src_curve[from_], marker='x', c="red", label="from_")
                            ax.plot(to_, src_curve[from_], marker='x', c="magenta", label="to_")

                    ## Show the plot
                    ax.legend(loc='best')
                    plt.xlabel('Time')
                    plt.ylabel('Amplitude')
                    plt.title(ss_seq_name_w_condition)
                    plt.tight_layout() 
                    # plt.grid()
                    plt.savefig(out_vis_file)
                    plt.savefig(out_vis_file.replace(".png", ".svg"))
                    print("Saved file", out_vis_file)  


                pose_pred_metric = outprocess_fn(pose_pred_normed_space) # in metric space
                if exp_cfg.render:
                ## visualize the results
                    curr_pred = self.face_model.apply_neck_rotation(exprs, pose_pred_metric[0]).detach()
                    out_file, pred_rendered_images = self.render_helper_obj.visualize_meshes(out_dir_video,
                                                                                            ss_seq_name_w_condition,
                                                                                            audio_file,
                                                                                            curr_pred,
                                                                                            "ss%02d" % i)
                    
                    cmb_file = out_file.replace(".mp4", "_comp.mp4")
                    comb_cmd = f"ffmpeg -y -i {gt_file} -i {out_file} -filter_complex hstack=inputs=2 {cmb_file}"
                    os.system(comb_cmd)
                    

                out_dict = {
                    "vertice" : exprs.detach().cpu().numpy(),
                    "pose" : pose_pred_metric[0].detach().cpu().numpy(),
                }
                
                ## dump the metric with head motion
                curr_out_file = os.path.join(out_dir_npy, ss_seq_name_w_condition+".npy")
                np.save(curr_out_file, out_dict)
                print("Dumping out file", curr_out_file)


def add_local_arguments(parser):
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=str, default=10)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--audio_file", type=str, default=None)
    parser.add_argument("--edit_file", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.set_defaults(unseen=False)
    return parser

if __name__ == "__main__":

    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model
    add_local_arguments(parser)
    args= parser.parse_args()

    exp_cfg_path = "face_animation_model/evaluate/complete_face_animation/cfg/comp_syn.yaml"
    exp_cfg = OmegaConf.load(exp_cfg_path)

    for k, v in exp_cfg.items():
        setattr(args, k, v)

    args.train_platform_type = "NoPlatform"
    fixseed(args.seed)

    # create the tester
    tester = test_w_sty_head()
    if args.audio_file is None:
        os.environ["HOME"] = "/home/bthambiraja"
        args.audio_file = "./assests/audio/01welcome.wav"
        # args.audio_file = "./assests/audio/k-pmfynqbko.wav"
        # args.audio_file = "./assests/audio/wD-jLNmRVfw.wav"

    if args.model_path is not None:
        args.head_motion_model.model_path = args.model_path
    
    if args.synthensis_mode == "generate":
        ## synthesis model
        assert args.audio_file is not None
        tester.run_external_audio(args)
    else:
        ### 
        assert args.audio_file is not None
        if args.edit_file is None:
            args.edit_file = args.exp_cfg.edit_file
        tester.run_edit_external_audio(args)
