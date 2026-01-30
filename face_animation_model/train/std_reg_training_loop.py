import copy
import functools
import os
import time
from types import SimpleNamespace
import numpy as np

import blobfile as bf
import torch
from torch.optim import AdamW

from diffusion import logger
from face_animation_model.utils import dist_util
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.resample import LossAwareSampler, UniformSampler
from tqdm import tqdm
from diffusion.resample import create_named_schedule_sampler
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from eval import eval_humanml, eval_humanact12_uestc
from data_loaders.get_data import get_dataset_loader


# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(self, args, train_platform, model, diffusion, data):
        self.args = args
        self.dataset = args.dataset
        self.train_platform = train_platform
        self.model = model
        self.diffusion = diffusion
        self.cond_mode = model.args.cond_mode
        self.data = data
        self.batch_size = args.batch_size
        self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval

        self.resume_checkpoint = args.resume_checkpoint
        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size # * dist.get_world_size()
        self.num_steps = args.num_steps
        if args.dataset == "vocaset":
            self.data_dict = self.data
            self.data = self.data._train_dataloader()

        self.num_epochs = self.num_steps // len(self.data) + 1

        self.sync_cuda = torch.cuda.is_available()
        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())

        # load the model
        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.save_dir = args.save_dir
        self.overwrite = args.overwrite

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        self.schedule_sampler_type = 'uniform'
        # self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None
        if args.dataset in ['kit', 'humanml'] and args.eval_during_training:
            mm_num_samples = 0  # mm is super slow hence we won't run it during training
            mm_num_repeats = 0  # mm is super slow hence we won't run it during training
            gen_loader = get_dataset_loader(name=args.dataset, batch_size=args.eval_batch_size, num_frames=None,
                                            split=args.eval_split,
                                            hml_mode='eval')

            self.eval_gt_data = get_dataset_loader(name=args.dataset, batch_size=args.eval_batch_size, num_frames=None,
                                                   split=args.eval_split,
                                                   hml_mode='gt')
            self.eval_wrapper = EvaluatorMDMWrapper(args.dataset, dist_util.dev())
            self.eval_data = {
                'test': lambda: eval_humanml.get_mdm_loader(
                    model, diffusion, args.eval_batch_size,
                    gen_loader, mm_num_samples, mm_num_repeats, gen_loader.dataset.opt.max_motion_length,
                    args.eval_num_samples, scale=1.,
                )
            }
        elif args.dataset in ['vocaset'] and args.eval_during_training:
            gen_loader = self.data_dict._val_dataloader()

        self.use_ddp = False
        self.ddp_model = self.model

        # added by bala on May 9th for training time visulization
        self.vis_interval = args.vis_interval
        from face_animation_model.visualizer.util_render_helper import render_helper
        self.render_helper = render_helper()
        self.train_out_dir = os.path.join(self.save_dir, "train_progress_vis")
        os.makedirs(self.train_out_dir, exist_ok=True)

        # added for the loss visulization
        self.mse_loss = torch.nn.MSELoss()

        self.const_noise = None
        from face_animation_model.diffusion.Disp_noise_sampler import constant_random_Noise_v2
        self.noise_sample = constant_random_Noise_v2()

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            result = self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                )
            )
            print("result of loading:\n", result)

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")

            # TODO: fix later by baala on 12-April
            # state_dict = dist_util.load_state_dict(
            #     opt_checkpoint, map_location=dist_util.dev()
            # )
            state_dict = torch.load(opt_checkpoint, map_location=self.device)
            result = self.opt.load_state_dict(state_dict)

            # added by bala to optimizer load issue
            for k, v in self.opt.state_dict()["state"].items():
                v["exp_avg"] = v["exp_avg"].to(self.device)
                v["exp_avg_sq"] = v["exp_avg_sq"].to(self.device)

            print("result of loading:\n", result)

    def run_loop(self):

        self.model.to(self.device)
        for epoch in range(self.num_epochs):
            print(f'Starting epoch {epoch}')
            for batch in tqdm(self.data):
                audio, motion, template, identity, filename = batch

                new_batch = []
                for x in batch:
                    if torch.is_tensor(x):
                        new_batch.append(x.to(self.device))
                    else:
                        new_batch.append(x)
                batch = new_batch

                if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                    break
                # cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}

                self.run_step(batch)
                if self.step % self.log_interval == 0:
                    for k,v in logger.get_current().name2val.items():
                        if k == 'loss':
                            print('step[{}]: loss[{:0.5f}]'.format(self.step+self.resume_step, v))

                        if k in ['step', 'samples'] or '_q' in k:
                            continue
                        else:
                            self.train_platform.report_scalar(name=k, value=v, iteration=self.step, group_name='Loss')

                if self.step % self.save_interval == 0:
                    # self.save()
                    self.model.eval()
                    # self.evaluate()
                    self.model.train()

                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                self.step += 1
            if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                break

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
            self.evaluate()

    def val_loop(self):

        # for data in the dataloader:
        # sample motion
        # compute the losses
        # report the losses
        pass

    def test_loop(self):

        # for data in the test loader
        # sample the motion
        # compute the losses and store as the results
        # visualize the results
        pass

    def run_step(self, batch):
        # self.forward_backward(batch)
        self.forward_backward_custom(batch)
        self.mp_trainer.optimize(self.opt)
        self._anneal_lr()
        self.log_step()

    def evaluate(self):
        if not self.args.eval_during_training:
            return

    def forward_backward_custom(self, batch):
        self.mp_trainer.zero_grad()
        audio, motion, template, identity, filename = batch
        template = template.unsqueeze(1)  # (1,1, V*3)
        disp = motion - template
        x_start = disp

        weights = 1.0
        t = torch.tensor([499]).long().to(motion.device)

        noise = self.const_noise
        # input
        x_t = x_start + noise
        noise_output = self.model(x_t, t, **{"batch": batch})

        losses = {}
        losses["rot_mse"] = self.mse_loss(noise, noise_output)  # mean_flat(rot_mse) # size [64]
        losses["loss"] = losses["rot_mse"] + losses.get('vb', 0.)

        # return model output
        losses["model_out"] = template + x_t - noise_output.clone().detach()
        model_out = losses.pop("model_out")  # only used for visulization

        loss = (losses["loss"] * weights).mean()
        self.mp_trainer.backward(loss)

        # visualize the prediction for every N batch
        if self.step % self.vis_interval == 0:
            self.visualize_during_train(t, model_out, batch)

    def visualize_during_train(self, diff_step, model_output, batch):

        # create the output folder
        Bs, T, feat_dim = model_output.shape
        audio, motion, template, identity, filename = batch
        # visualize only the first sequence
        curr_pred = model_output[0].reshape(T, -1, 3) # T x 5023 x 3
        curr_gt = motion[0].reshape(T, -1, 3) # T x 5023 x 3

        curr_name = filename[0].split(".")[0] + "_step_%05d"%self.step + "_diffstep_%05d"%diff_step[0]
        _, _ = self.render_helper.visualize_meshes_with_gt(self.train_out_dir,
                                                            curr_name,
                                                            None, # audio file
                                                            curr_pred,
                                                           curr_gt
                                                    )

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def ckpt_file_name(self):
        return f"model{(self.step+self.resume_step):09d}.pt"

    def save(self):
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)

            # Do not save CLIP weights
            clip_weights = [e for e in state_dict.keys() if e.startswith('clip_model.')]
            for e in clip_weights:
                del state_dict[e]

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, "checkpoints", filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)

        with bf.BlobFile(
            bf.join(self.save_dir, "checkpoints", f"opt{(self.step+self.resume_step):09d}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
