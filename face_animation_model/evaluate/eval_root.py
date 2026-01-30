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

# os.environ['PYOPENGL_PLATFORM'] = 'egl'


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


class test():

    def __init__(self,
                num_seq_to_render = 20,
                num_frames_to_render = 450,
                visualizer_cfg=None,
                data_cfg=None,
                **kwargs) -> None:
        self.num_seq_to_render = num_seq_to_render
        self.num_frames_to_render = num_frames_to_render
        self.visualizer_cfg = visualizer_cfg
        self.dataset = data_cfg.dataset
        self.data_cfg = data_cfg

        if self.visualizer_cfg is None:
            self.set_default_visualizer(data_cfg)
        else:
            self.visualizer_cfg.params.data_cfg = data_cfg
            self.face_visualzer = init_from_config(self.visualizer_cfg)

    def set_default_visualizer(self, data_cfg):
        from face_animation_model.visualizer.visualize_faceformer_full_training import face_visulizer_voca_faceformer
        self.face_visualzer = face_visulizer_voca_faceformer(aux_curves=False,
                                                             data_cfg=data_cfg)

    def eval_loop(self, args, data, diffusion_model, model):

        results_dict_list = []
        loss = []
        loss_in_mm = []
        for batch in data:
            result_npy_dict = {}

            new_batch = []
            for x in batch:
                if torch.is_tensor(x):
                    new_batch.append(x.to(args.device))
                else:
                    new_batch.append(x)
            batch = new_batch

            sample_fn = diffusion_model.p_sample_loop
            audio, motion, template, one_hot, file_name = batch
            seq_len = (motion.shape[1] // 8) * 8
            motion = motion[:, :seq_len]
            shape = motion.shape

            prediction = sample_fn(model, shape, model_kwargs={"batch":batch})

            # compute the losses
            loss.append(model.loss(prediction, motion).item())
            loss_in_mm.append(model.loss(prediction * 1000.0, motion * 1000.0).item())

            # simple metric rec loss
            loss_dict = {
                "metric_rec_loss": np.mean(loss),
                "metric_rec_loss_in_mm": np.mean(loss_in_mm),
            }

            # out_file = file_name[0].split(".")[0] + "_condition_" + train_subject
            out_file = file_name[0].split(".")[0]
            prediction = prediction.squeeze()
            result_npy_dict[out_file] = prediction.detach().cpu().numpy()

            gt_kp = motion.cpu().numpy().squeeze()
            out_dict = {'results': loss_dict,
                        'prediction_dict': result_npy_dict,
                        "gt_kp": gt_kp,
                        'seq_name': file_name,
                        'seq_len': prediction.shape[0]
                        }

            results_dict_list.append(out_dict)

        return results_dict_list

    def run_inference_for_dataset(self, args, logdir, dataloader, mode,
                        inference_model, diffusion_model, post_fix=None):

            inference_model.to(args.device)
            if post_fix is None:
                mode_outdir = os.path.join(logdir, mode)
            else:
                mode_outdir = os.path.join(logdir, mode + post_fix)

            os.makedirs(mode_outdir, exist_ok=True)

            print()
            print('Running test on ', mode_outdir)
            # setting the inference model to eval
            inference_model = inference_model.eval()
            results_dict = self.eval_loop(args, dataloader, diffusion_model, inference_model)

            # store the losses
            losses = process_result_dict(results_dict)
            with open(os.path.join(mode_outdir, 'results.json'), 'w') as file:
                file.write(json.dumps(losses, indent=4))

            # storing the keypoints and rendering visualziations
            outfile = os.path.join(mode_outdir, "results_dict.npy")
            np.save(outfile, results_dict)
            print("Outfile for the results:", outfile)
            # get visualizations
            self.face_visualzer.run_render_results(mode_outdir, number_seq=self.num_seq_to_render, num_render_frames=self.num_frames_to_render)

            return losses

    def run_extensive_test(self, args, model, diffusion_model, data, logdir):

        combnd_result = {}

        # adding total trainable parameters
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        combnd_result["trainable_total_params"] = pytorch_total_params

        # load the best checkpoint based on the validation results and  store the result

        if hasattr(args, "model_ckpt"):
            best_ckpt = os.path.join(logdir, "checkpoints", args.model_ckpt + ".pt")
        else:
            best_ckpt = get_latest_checkpoint(os.path.join(logdir, "checkpoints"))
            print("Current best checkpoint", best_ckpt)
        
        model.init_from_ckpt(path=best_ckpt)

        if data._test_dataloader() is not None:
            combnd_result["test"] = self.run_inference_for_dataset(args, logdir, data._test_dataloader(), "test",
                                            model, diffusion_model)

        if data._val_dataloader() is not None:
            combnd_result["val"] = self.run_inference_for_dataset(args, logdir, data._val_dataloader(), "val",
                                            model, diffusion_model)

        with open(os.path.join(logdir, 'combnd_results.json'), 'w') as file:
            file.write(json.dumps(combnd_result, indent=4))


def parse_and_load_from_model(parser):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    args = parser.parse_args()
    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    return args

def add_local_arguments(parser):
    parser.add_argument("--gpus", type=str, default=0)
    parser.add_argument("--model_ckpt", type=str, default=None)
    parser.set_defaults(unseen=False)
    return parser


if __name__ == "__main__":

    parser = ArgumentParser()

    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_evaluation_options(parser)
    add_local_arguments(parser)
    args=parse_and_load_from_model(parser)

    cfg_path = os.path.join(args.model_path, 'args.yml')
    cfg = OmegaConf.load(cfg_path)
    # replace the default with the cfg
    for k, v in cfg.items():
        setattr(args, k, v)

    args.train_platform_type = "NoPlatform"
    model_path = args.model_path
    print("Model path", args.model_path)
    args.save_dir = args.model_path

    # create the model name
    # exp_name = args.cfg.split("/")[-1].split(".")[0]
    # save_dir = os.path.join(args.save_dir, exp_name)
    # args.save_dir = save_dir
    # args.log_dir = os.path.join(save_dir, "log")

    fixseed(args.seed)

    print("creating data loader...")
    data = init_from_config(args.data_cfg)

    print("creating model and diffusion...")
    model = init_from_config(args.motion_model)

    print("creating the diffusion model")
    # diffusion = init_from_config(args.diffusion)
    if cfg.get("diffusion") is not None:
        print("creating the diffusion model")
        diffusion = init_from_config(args.diffusion)
    else:
        print("diffusion model is None")
        diffusion = None

    # create the tester
    tester = test(args, data_cfg=args.data_cfg)
    tester.run_extensive_test(args, model, diffusion, data, args.save_dir)