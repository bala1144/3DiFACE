from face_animation_model.train.training_loop import *
import matplotlib.pyplot as plt
plt.switch_backend('agg')

class Head_reg_TrainLoop(TrainLoop):
    def __init__(self, **ignore_args):
        super().__init__(**ignore_args)

    def set_attributes(self, args, train_platform, model, diffusion, data):
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

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None
        if args.dataset in ['vocaset'] and args.eval_during_training:
            gen_loader = self.data_dict._val_dataloader()

        self.use_ddp = False
        self.ddp_model = self.model

        # added by bala on May 9th for training time visulization
        self.vis_interval = args.vis_interval
        from face_animation_model.visualizer.util_render_helper import render_helper
        self.render_helper = render_helper()
        self.train_out_dir = os.path.join(self.save_dir, "train_progress_vis")
        os.makedirs(self.train_out_dir, exist_ok=True)

        #### creating the flame model for loss
        config = {}
        if len(config) == 0:
            config["flame_model_path"] = os.path.join(os.getenv('HOME'),
                                                      "projects/motion_root/FLAMEModel",
                                                      "model/generic_model.pkl")
            config["batch_size"] = 1
            config["shape_params"] = 0
            config["expression_params"] = 0
            config["pose_params"] = 0
            config["number_worker"] = 8
            config["use_3D_translation"] = False

        from FLAMEModel.FLAME import FLAME
        self.face_model = FLAME(config)
        self.face_model = self.face_model.to(self.device)


    def forward_backward_custom(self, batch):
        self.mp_trainer.zero_grad()
        audio, motion, template, identity, filename, gt_pose = batch
        
        # takes to gt_pose

        t, weights = self.schedule_sampler.sample(motion.shape[0], dist_util.dev())
        new_batch = (audio, motion, template, identity, filename, gt_pose)
        losses = self.diffusion.training_losses(self.model, new_batch, t)
        model_out = losses.pop("model_out")
        
        # x_t
        if "x_t" in losses.keys():
            x_t = losses.pop("x_t")
        else:
            x_t = None

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses["loss"].detach()
            )

        loss = (losses["loss"] * weights).mean()
        log_loss_dict(
            self.diffusion, t, {k: v * weights for k, v in losses.items() if k != "model_out"}
        )
        self.mp_trainer.backward(loss)

        # visualize the prediction for every N batch
        if self.step % self.vis_interval == 0:
            self.visualize_during_train(t, model_out, batch, x_t)

    def visualize_during_train(self, diff_step, model_output, batch,  x_t = None):

        # # create the output folder
        # Bs, T, feat_dim = model_output.shape
        # audio, motion, template, identity, filename = batch
        # # visualize only the first sequence
        # curr_pred = model_output[0].reshape(T, -1, 3) # T x 5023 x 3
        # curr_gt = motion[0].reshape(T, -1, 3) # T x 5023 x 3
        #
        # curr_name = filename[0].split(".")[0] + "_step_%05d"%self.step + "_diffstep_%05d"%diff_step[0]
        # _, _ = self.render_helper.visualize_meshes_with_gt(self.train_out_dir,
        #                                                     curr_name,
        #                                                     None, # audio file
        #                                                     curr_pred,
        #                                                    curr_gt
        #                                             )

        # create the output folder
        Bs, T, feat_dim = model_output.shape
        audio, motion, template, identity, filename, pose = batch

        curr_gt = self.face_model.apply_neck_rotation(motion[0], pose[0]).cpu()
        curr_pred = self.face_model.apply_neck_rotation(motion[0], model_output[0]).cpu()
        # visualize only the first sequence
        curr_pred = curr_pred.reshape(T, -1, 3)  # T x 5023 x 3
        curr_gt = curr_gt.reshape(T, -1, 3)  # T x 5023 x 3

        curr_name = filename[0].split(".")[0] + "_step_%05d" % self.step + "_diffstep_%05d" % diff_step[0]
        # _, _ = self.render_helper.visualize_meshes_with_gt(self.train_out_dir,
        #                                                    curr_name,
        #                                                    None,  # audio file
        #                                                    curr_pred,
        #                                                    curr_gt
        #                                                    )

        fig, ax = plt.subplots(figsize=(15,6))
        time = np.linspace(0, 1, T)

        curr_pred = model_output[0].reshape(T, -1).cpu()  # T x 3
        curr_gt = pose[0].reshape(T, -1).cpu()  # T x 3

        ax.plot(time, torch.mean(curr_gt, -1),  color='red', label="gt")  # Adjust alpha for transparency

        ### head motion visulization
        if x_t is not None:
            curr_xt = x_t[0,:,:-1].detach().cpu() # T x 3
            ax.plot(time, torch.mean(curr_xt, -1),  color='orange', label="x_t")  # Adjust alpha for transparency

            # if x_t.shape[-1] in [2, 4, 7]:
            #     curr_mask = x_t[0,:,-1].detach().cpu() # T x 1
            #     ax.plot(time, curr_mask,  color='magenta', label="mask")  # Adjust alpha for transparency

        ax.plot(time, torch.mean(curr_pred, -1),  color='blue', label="pred")  # Adjust alpha for transparency
        ax.legend(loc='best')
        ## Show the plot
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title('Train diff step %03d'%diff_step[0])
        plt.tight_layout() 
        out_vis_file = os.path.join(self.train_out_dir, curr_name+".png")
        plt.savefig(out_vis_file)
        print("Saved file", out_vis_file)  
        plt.close()

        print("Done")

