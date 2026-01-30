from face_animation_model.evaluate.eval_root import *
from tqdm import tqdm

class test_w_sty(test):

    def __init__(self,
                **kwargs) -> None:
        super().__init__(**kwargs)
        # super().__init__(num_seq_to_render, num_frames_to_render, visualizer_cfg, dataset, data_cfg, HQ, ratio_800x1200=ratio_800x1200)
        # setting the condition to visulize for explict testing
        self.condition_id_to_visualize = self.data_cfg.condition_id_to_visualize.split(" ")

    def eval_loop(self, args, data, diffusion_model, model):

        # print("Setting model to train for debug")
        # model.train()

        results_dict_list = []
        loss = []
        loss_in_mm = []
        for batch in tqdm(data, desc="iterating through the batch"):

            new_batch = []
            for x in batch:
                if torch.is_tensor(x):
                    new_batch.append(x.to(args.device))
                else:
                    new_batch.append(x)
            batch = new_batch

            sample_fn = diffusion_model.p_sample_loop
            if len(batch) == 5:
                audio, motion, template, one_hot_input, file_name = batch
            else:
                audio, motion, template, one_hot_input, file_name, gt_pose = batch

                if model.args.in_channel in [3,4, 6, 7]: # check for head pose
                    motion = gt_pose

            nf = motion.shape[1]
            if nf != 30:
                nf = (nf // 8) * 8  # enable compression
                motion = motion[:, :nf]

                if len(audio.shape) == 2:
                    pass
                else:
                    audio = audio[:, :nf]

            for test_subj_name in self.condition_id_to_visualize:
                
                shape = motion.shape
                result_npy_dict = {}

                # generate the one hot encoding for the model
                if len(model.train_subjects) > 1:
                    index = model.train_subjects.index(test_subj_name)
                    # one_hot = torch.zeros((1, len(model.train_subjects)), dtype=one_hot_input.dtype, device=audio.device)
                    one_hot = torch.zeros((1, model.args.num_identity_classes), dtype=one_hot_input.dtype, device=audio.device)
                    one_hot[0, index] = 1
                else:
                    # used for the style adaption
                    one_hot = one_hot_input
                    test_subj_name = model.train_subjects[0]

                # print(test_subj_name, index, one_hot.shape)
                new_batch = (audio, motion, template, one_hot, file_name) 
                print(shape, audio.shape, motion.shape, file_name)
                prediction = sample_fn(model, shape, model_kwargs={"batch":new_batch})

                # compute the losses
                loss.append(model.loss(prediction, motion).item())
                loss_in_mm.append(model.loss(prediction * 1000.0, motion * 1000.0).item())

                # simple metric rec loss
                loss_dict = {
                    "metric_rec_loss": np.mean(loss),
                    "metric_rec_loss_in_mm": np.mean(loss_in_mm),
                }

                out_file = file_name[0].split(".")[0] + "_condition_" + test_subj_name
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

class test_w_sty_blendvoca(test):

    def __init__(self,
                **kwargs) -> None:
        super().__init__(**kwargs)
        # super().__init__(num_seq_to_render, num_frames_to_render, visualizer_cfg, dataset, data_cfg, HQ, ratio_800x1200=ratio_800x1200)
        # setting the condition to visulize for explict testing
        self.condition_id_to_visualize = self.data_cfg.condition_id_to_visualize.split(" ")

        config = {}
        if len(config) == 0:
            config["flame_model_path"] = os.path.join(os.getenv("HOME"),
                                                      "projects/dataset/BlendVOCA")
            config["batch_size"] = 1
        ## this is not not needed for the BIWi model
        from FLAMEModel.FLAME_Blend import FLAME
        self.face_model = FLAME(config)

        from face_animation_model.visualizer.util_render_helper import render_helper
        self.render_helper_obj = render_helper(dataset=self.dataset, face_model=self.face_model)
        self.face_model = self.face_model.to(kwargs.get("device", "cpu"))

        self.num_seq_to_render = kwargs.get("num_seq_to_render", 20)
        self.dump_npy = kwargs.get("dump_npy", True)
        self.render_vid = kwargs.get("render_vid", True)
        
    def eval_loop(self, args, data, diffusion_model, model, out_dir):
    
        if self.dump_npy:
            npy_dir = os.path.join(out_dir, "npy_dump")
            os.makedirs(npy_dir, exist_ok=True)

        # print("Setting model to train for debug")
        # model.train()
        results_dict_list = []
        loss = []
        loss_in_mm = []
        num_render_seq = 0
        for batch in tqdm(data, desc="iterating through the batch"):

            new_batch = []
            for x in batch:
                if torch.is_tensor(x):
                    new_batch.append(x.to(args.device))
                else:
                    new_batch.append(x)
            batch = new_batch

            sample_fn = diffusion_model.p_sample_loop
            if len(batch) == 5:
                audio, bcoeffs, template, one_hot_input, file_name = batch
            else:
                audio, bcoeffs, template, one_hot_input, file_name, gt_pose = batch

                if model.args.in_channel in [3,4, 6, 7]: # check for head pose
                    bcoeffs = gt_pose

            nf = bcoeffs.shape[1]
            if nf != 30:
                nf = (nf // 8) * 8  # enable compression
                bcoeffs = bcoeffs[:, :nf]

                if len(audio.shape) == 2:
                    pass
                else:
                    audio = audio[:, :nf]

            for test_subj_name in self.condition_id_to_visualize:
                
                shape = bcoeffs.shape
                result_npy_dict = {}

                # generate the one hot encoding for the model
                if len(model.train_subjects) > 1:
                    index = model.train_subjects.index(test_subj_name)
                    # one_hot = torch.zeros((1, len(model.train_subjects)), dtype=one_hot_input.dtype, device=audio.device)
                    one_hot = torch.zeros((1, model.args.num_identity_classes), dtype=one_hot_input.dtype, device=audio.device)
                    one_hot[0, index] = 1
                else:
                    # used for the style adaption
                    one_hot = one_hot_input
                    test_subj_name = model.train_subjects[0]

                # print(test_subj_name, index, one_hot.shape)
                new_batch = (audio, bcoeffs, template, one_hot, file_name) 
                print(shape, audio.shape, bcoeffs.shape, file_name)
                bcoeffs_pred = sample_fn(model, shape, model_kwargs={"batch":new_batch})
                disp, disp_pred = self.face_model.morph_with_gt(bcoeffs, bcoeffs_pred, file_name)
                prediction = disp_pred + template
                motion = disp + template

                # compute the losses
                loss.append(model.loss(prediction, motion).item())
                loss_in_mm.append(model.loss(prediction * 1000.0, motion * 1000.0).item())

                # simple metric rec loss
                loss_dict = {
                    "metric_rec_loss": np.mean(loss),
                    "metric_rec_loss_in_mm": np.mean(loss_in_mm),
                }

                seq_name_with_condition = file_name[0].split(".")[0] + "_condition_" + test_subj_name
                prediction = prediction.squeeze()
                result_npy_dict[seq_name_with_condition] = prediction.detach().cpu().numpy()


                gt_kp = motion.cpu().numpy().squeeze()
                out_dict = {'results': loss_dict,
                            'prediction_dict': result_npy_dict,
                            "gt_kp": gt_kp,
                            'seq_name': file_name,
                            'seq_len': prediction.shape[0]
                            }
                results_dict_list.append(out_dict)

                    ## dump the prediction
                if self.dump_npy:
                # if True:
                    subj = file_name[0].split("_sent")[0]
                    sent = file_name[0].split("_TA_")[-1]
                    audio_file = os.path.join(os.getenv("HOME"), args.data_cfg.params.dataset_root, "wav", subj, sent)
                    curr_out_file = os.path.join(npy_dir, seq_name_with_condition+".npy")
                    out_dict = {
                            "gt": gt_kp,
                            'template':  template.unsqueeze(0).cpu().numpy(),
                            'audio_file': audio_file,
                            'prediction': prediction.detach().cpu().numpy(),
                            'bcoeffs_pred': bcoeffs_pred.detach().cpu().numpy(),
                            'bcoeffs': bcoeffs.detach().cpu().numpy(),
                    }
                    # np.save(curr_out_file, prediction.detach().cpu().numpy())
                    np.save(curr_out_file, out_dict)
                    print("Dumping out file", curr_out_file)

                # add visualization
                if num_render_seq < self.num_seq_to_render and self.render_vid: 
                    # gt
                    gt_motion = motion.reshape(motion.shape[1], -1, 3).detach()
                    gt_file, _ = self.render_helper_obj.visualize_meshes(out_dir, 
                                                                         "gt",
                                                                        None,
                                                                        gt_motion,
                                                                        annotation="GT")
                    
                    #   pred
                    curr_pred = prediction.reshape(motion.shape[1], -1, 3).detach()
                    pred_file, _ = self.render_helper_obj.visualize_meshes(out_dir,
                                                                            "pred",
                                                                            None,
                                                                            curr_pred,
                                                                            annotation="Pred")
                    
                    # compute the diffrence
                    # import pdb; pdb.set_trace()
                    diff = template.unsqueeze(1) + torch.abs(prediction - motion)
                    diff = diff.reshape(motion.shape[1], -1, 3).detach()
                    diff_file, _ = self.render_helper_obj.visualize_meshes(out_dir,
                                                                        "diff",
                                                                            None,
                                                                            diff,
                                                                            annotation="Diff")
                    
                    
                    # combine the
                    out_file = os.path.join(out_dir, seq_name_with_condition + ".mp4")
                    os.system(f'ffmpeg -y -i {gt_file} -i {pred_file} -i {diff_file} -filter_complex "hstack=inputs=3[out]" -map "[out]" {out_file}')
                    # add audio to video
                    out_file_w_aud = out_file.replace(".mp4", "_waud.mp4")
                    subj = file_name[0].split("_sent")[0]
                    sent = file_name[0].split("_TA_")[-1]
                    audio_file = os.path.join(os.getenv("HOME"), args.data_cfg.params.dataset_root, "wav", subj, sent)
                    self.render_helper_obj.add_audio_to_video(audio_file, out_file, out_file_w_aud)
                    
                    os.system(f"rm {gt_file} {pred_file} {diff_file}")

                    num_render_seq+=1


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
            results_dict = self.eval_loop(args, dataloader, diffusion_model, inference_model,mode_outdir )

            # store the losses
            losses = process_result_dict(results_dict)
            with open(os.path.join(mode_outdir, 'results.json'), 'w') as file:
                file.write(json.dumps(losses, indent=4))
            return losses

    def run_extensive_test_local(self, args, model, diffusion_model, data, logdir):

        combnd_result = {}

        # adding total trainable parameters
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        combnd_result["trainable_total_params"] = pytorch_total_params

        # load the best checkpoint based on the validation results and  store the result

        if args.model_ckpt is not None:
            best_ckpt = os.path.join(logdir, "checkpoints", args.model_ckpt + ".pt")
        else:
            best_ckpt = get_latest_checkpoint(os.path.join(logdir, "checkpoints"))
            print("Current best checkpoint", best_ckpt)
        model.init_from_ckpt(path=best_ckpt)

        if data._test_dataloader() is not None:
            combnd_result["test"] = self.run_inference_for_dataset(args, logdir, data._test_dataloader(), "test",
                                            model, diffusion_model)

        # if data._val_dataloader() is not None:
        #     combnd_result["val"] = self.run_inference_for_dataset(args, logdir, data._val_dataloader(), "val",
        #                                     model, diffusion_model)

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
    diffusion = init_from_config(args.diffusion)

    # create the tester
    # tester = test_w_sty(args, data_cfg=args.data_cfg)
    # tester.run_extensive_test(args, model, diffusion.diffusion_model, data, args.save_dir)

    ## code for the blendvoca
    if args.model_ckpt is not None:
        cfg.model_ckpt = args.model_ckpt
        print(f"\n Resetting model to test time model ckpt {cfg.model_ckpt}")

    tester = test_w_sty_blendvoca(args=args, data_cfg=args.data_cfg)
    # tester.run_extensive_test(args, model, diffusion, data, args.save_dir)
    tester.run_extensive_test_local(args, model, diffusion, data, args.save_dir)

