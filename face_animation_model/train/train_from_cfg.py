# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
from motion_diffusion_model.utils.fixseed import fixseed
from motion_diffusion_model.utils.parser_util import train_args
from motion_diffusion_model.utils import dist_util
from face_animation_model.train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation
from omegaconf import OmegaConf
from face_animation_model.utils.init_helper import init_from_config

def main():
    args = train_args()
    cfg = OmegaConf.load(args.cfg)
    # replace the default with the cfg
    for k, v in cfg.items():
        setattr(args, k, v)

    # create the model name
    exp_name = args.cfg.split("/")[-1].split(".")[0]
    save_dir = os.path.join(args.save_dir, exp_name)
    print("Saving dir", save_dir)
    os.makedirs(save_dir, exist_ok=True)
    args.save_dir = save_dir
    args.log_dir = os.path.join(save_dir, "log")
    os.environ["OPENAI_LOGDIR"] = args.log_dir

    fixseed(args.seed)
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.log_dir)
    train_platform.report_args(args, name='Args')

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.yml')
    OmegaConf.save(vars(args), args_path)
    dist_util.setup_dist(args.device)

    print("creating data loader...")
    # data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames)
    data = init_from_config(args.data_cfg)

    print("creating model...")
    dist_util.setup_dist()
    model = init_from_config(args.motion_model)
    model.to(dist_util.dev())

    if cfg.get("diffusion") is not None:
        print("creating the diffusion model")
        diffusion = init_from_config(args.diffusion)
 
    else:
        print("diffusion model is None")
        diffusion = None
    # # added this code for legacy support
    # if not hasattr(diffusion, "diffusion_model"):
    #     diffusion_model = diffusion

    from scripts.src_bkp import take_snap_shot
    take_snap_shot(args.save_dir)

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training...")

    # # trainer = TrainLoop(args, train_platform, model, diffusion.diffusion_model, data)
    # if diffusion is not None:
    #     # call the diffusion trainer
    #     print("Running Diffusion trainer")
    #     from face_animation_model.mdm_train.training_loop import TrainLoop
    #     trainer = TrainLoop(args, train_platform, model, diffusion, data)
    # else:
    #
    #     # replace with the below loop
    #     # from face_animation_model.mdm_train.std_reg_training_loop import TrainLoop
    #     # trainer = TrainLoop(args, train_platform, model, diffusion, data)
    #     trainer = init_from_config(args.train_loop_cfg)
    #     trainer.set_attributes(args, train_platform, model, diffusion, data)

    #### training the diffusion and standard reg model
    trainer = init_from_config(args.train_loop_cfg)
    trainer.set_attributes(args, train_platform, model, diffusion, data)

    trainer.run_loop()
    train_platform.close()

    # run the evaluation
    eval_cfg = args.eval_cfg
    eval_cfg.params.data_cfg = args.data_cfg
    eval_cfg.params.device = args.device
    tester = init_from_config(eval_cfg)
    tester.run_extensive_test(args, model, diffusion, data, args.save_dir)


if __name__ == "__main__":
    main()
