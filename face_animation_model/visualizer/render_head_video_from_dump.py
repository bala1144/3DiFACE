from face_animation_model.evaluate.eval_root import *
from face_animation_model.utils.fixseed import fixseed
import scipy
from scipy.spatial.transform import Rotation
# from data.hdtf_dataset.debug_data_loader_faster_collate import pose_processing
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
title_font = {'fontname': 'DejaVu Sans', 'size': '16', 'color': 'black', 'weight': 'normal',
              'verticalalignment': 'bottom'}

import torch
import numpy as np
import os, datetime, glob, importlib
import subprocess
from tqdm import tqdm
import cv2
from face_animation_model.visualizer.util_pyrenderer import Facerender
from face_animation_model.visualizer.util_pyrenderer_final_result_version import Facerender as FHQ
from FLAMEModel.FLAME import FLAME
from glob import glob
from skimage import color
import pickle
from face_animation_model.utils.torch_rotation import *

class render_helper():
    def __init__(self, config = {}, render_type="std"):
        if len(config) == 0:
            config["flame_model_path"] = os.path.join(os.getenv('HOME'),
                                                      "projects/NeuralMotionSynthesis/FLAMEModel",
                                                      "model/generic_model.pkl")
            config["batch_size"] = 1
            config["shape_params"] = 0
            config["expression_params"] = 0
            config["pose_params"] = 0
            config["number_worker"] = 8
            config["use_3D_translation"] = False

        from FLAMEModel.FLAME import FLAME
        self.face_model = FLAME(config)
        self.cust_trans = np.zeros((1, 3))

        if render_type == "paper_quality":
            self.image_size = (800, 800)
            self.face_render = FHQ()
        
        elif render_type == "ratio_800x1200":
            self.image_size = (800, 1200)
            self.face_render = FHQ(img_size=(800, 1200))
            self.cust_trans[0, 2] = 1.35
            self.cust_trans[0, 1] = 0.01
        
        elif render_type == "custom_camera":
            self.image_size = (800, 1200)
        
            # face_render3
            camera_rotation = np.eye(4)
            camera_rotation[:3, :3] = Rotation.from_euler('z', 0, degrees=True).as_matrix() @ Rotation.from_euler('y', 0, degrees=True).as_matrix() @ Rotation.from_euler(
                'x', -15, degrees=True).as_matrix()
            camera_translation = np.eye(4)
            camera_translation[:3, 3] = np.array([0, 0, 1])
            camera_pose3 = camera_rotation @ camera_translation
            self.face_render = FHQ(img_size=self.image_size, camera_pose=camera_pose3)
        
            self.cust_trans[0, 1] = 0.05
            self.cust_trans[0, 2] = 0.20

        else:
            self.image_size = (512, 512)
            from face_animation_model.visualizer.util_pyrenderer import Facerender
            self.face_render = Facerender()

    def dist_to_rgb(self, errors, min_dist=0.0, max_dist=1.0):
        import matplotlib as mpl
        import matplotlib.cm as cm
        norm = mpl.colors.Normalize(vmin=min_dist, vmax=max_dist)
        cmap = cm.get_cmap(name='jet')
        colormapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        return colormapper.to_rgba(errors)[:, 0:3]

    def error_in_mm(self, pred_verts, gt_verts, vertice_dim=15069):
        pred_verts_mm = pred_verts.view(-1, vertice_dim // 3, 3) * 1000.0
        gt_verts_mm = gt_verts.view(-1, vertice_dim // 3, 3) * 1000.0
        diff_in_mm = pred_verts_mm - gt_verts_mm
        dist_in_mm = torch.sqrt(torch.sum(diff_in_mm ** 2, dim=-1))
        return torch.mean(dist_in_mm, dim=0)  # 5023 x 1

    def render_heat_map(self, out_dir, out_seq_name,
                        gt, pred, template, max_error=10):
        out_pred_rendered_images = []
        vid_frames = []
        for i in tqdm(range(pred.shape[0]), desc="Rendering heat map"):
            error = self.error_in_mm(pred[i],
                                     gt[0, i])
            colours = self.dist_to_rgb(error.cpu().numpy(),
                                       0,
                                       max_error)
            pred_frame = self.face_render.render_heat_map(pred[i].cpu().numpy(),
                                                          self.face_model.faces,
                                                          colours)
            out_pred_rendered_images.append(np.copy(pred_frame))

            cv2.putText(pred_frame, f"max {max_error}mm", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            vid_frames.append(pred_frame)


        out_vid_file = os.path.join(out_dir, out_seq_name + "_eval_nf%s.mp4" % len(vid_frames))
        video_file = self.compose_write_video(out_vid_file, vid_frames, self.image_size)

        return video_file, out_pred_rendered_images

    def render_meshes(self, pred_vertices):
        out_pred_rendered_images = []
        for i in tqdm(range(pred_vertices.shape[0]), desc="Rendering expressions"):
            pred_frame = self.render_exprs_to_image(pred_vertices[i])
            out_pred_rendered_images.append(pred_frame)
        return out_pred_rendered_images

    def visualize_meshes(self, out_dir, out_seq_name, audio_file, pred_vertices, desc="expression"):

        out_pred_rendered_images = []
        for i in tqdm(range(pred_vertices.shape[0]), desc=f"Rendering {desc}"):
            pred_frame = self.render_exprs_to_image(pred_vertices[i])

            if "expression" != desc and desc is not None:
                cv2.putText(pred_frame, desc, (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)
            out_pred_rendered_images.append(pred_frame)

        # create the video out file
        out_vid_file = os.path.join(out_dir, out_seq_name + "_eval_nf%s.mp4" % len(out_pred_rendered_images))
        video_file = self.compose_write_video(out_vid_file, out_pred_rendered_images, self.image_size)

        if audio_file is not None:
            video_out=video_file.replace(".mp4", "_aud.mp4")
            self.add_audio_to_video(audio_file, video_file, video_out)
            video_file = video_out

        return video_file, out_pred_rendered_images

    def visualize_meshes_with_kf(self, out_dir, out_seq_name, pred_vertices, key_frame = [], kf_string="keyframe"):

        out_pred_rendered_images = []
        for i in tqdm(range(pred_vertices.shape[0]), desc="Rendering expressions"):
            pred_frame = self.render_exprs_to_image(pred_vertices[i])
            cv2.putText(pred_frame, f"in_motion", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                        cv2.LINE_AA)
            if i in key_frame or (i-pred_vertices.shape[0]) in key_frame:
                id = "%03d"%i
                cv2.putText(pred_frame, f"f:{id} {kf_string}", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                            cv2.LINE_AA)
            out_pred_rendered_images.append(pred_frame)


        # create the video out file
        out_vid_file = os.path.join(out_dir, out_seq_name + "_eval_nf%s.mp4" % len(out_pred_rendered_images))
        video_file = self.compose_write_video(out_vid_file, out_pred_rendered_images, self.image_size)

        return video_file, out_pred_rendered_images

    def write_images(self, pred_images_folder, seq_name, pred_frames):
        image_out_folder = os.path.join(pred_images_folder, seq_name)
        os.makedirs(image_out_folder, exist_ok=True)
        print('Writing Images to', image_out_folder)
        for i, frame in tqdm(enumerate(pred_frames), desc="writing pred images"):
            outfile = os.path.join(image_out_folder, "frame%04d.jpg" % i)
            cv2.imwrite(outfile, frame)

    def render_exprs_to_image(self, exprs, shape_params=None):
        """
        exprs:  N x 103 (jawpose, exprs)
        """
        if exprs.shape[0] < 200:
            # print("exprs before pose", exprs.shape)
            jaw_pose = exprs[:3].view(1,-1)
            exprs_only = exprs[3:].view(1,-1)
            # print("Jaw pose", jaw_pose.shape)
            # print("exprs_only pose", exprs_only.shape)
            # print("shape_params", shape_params.shape if shape_params is not None else None)
            flame_vertices = self.face_model.morph(expression_params=exprs_only, jaw_pose=jaw_pose, shape_params=shape_params)[0]
        else:
            flame_vertices = exprs.reshape(-1,3)

        rendered_frame = self.render_images(flame_vertices)
        return rendered_frame

    def render_images(self, vertices):
        # import pdb; pdb.set_trace()
        vertices = vertices.cpu().numpy() + self.cust_trans
        self.face_render.add_face(vertices, self.face_model.faces)
        colour = self.face_render.render()
        return colour

    def compose_write_video(self, out_vid_file, gt_frames,
                            frame_size=(512, 512), fps=30):

        print('Writing video', out_vid_file)
        print()
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        writer = cv2.VideoWriter(out_vid_file, fourcc, fps, frame_size)
        # print("diff frames", len(diff_frames))
        for i, frame in tqdm(enumerate(gt_frames), desc="writing video"):
            out_frame = frame
            writer.write(out_frame)
        writer.release()

        return out_vid_file

    def add_audio_to_video(self, audio_file, video_file, out_vid_w_audio_file):

        ffmpeg_command = f"ffmpeg -y -i {video_file} -i {audio_file} -map 0:v -map 1:a -c:v copy -shortest {out_vid_w_audio_file}"
        print("ffmpeg_command", ffmpeg_command)
        os.system(ffmpeg_command)

        rm_command = ('rm {0} '.format(video_file))
        out = subprocess.call(rm_command.split())
        print("removed ", video_file)

class head_render():

    def __init__(self, render_type="std") -> None:
        self.render = render_helper(render_type=render_type)

    def render_from_dump(model_path):

        print("Loading the pred files")
        all_files = sorted(glob(os.path.join(model_path, "*.npy")))
        print("all the files to process", len(all_files))

        max_seq_len=576
        pred_poses = {}
        for pred_file in all_files:
            pose = np.load(pred_file, allow_pickle=True)[0] # T x 3
            seq_name = pred_file.split("/")[-1].split(".")[0]
            pred_poses[seq_name] = pose[:max_seq_len]
            break

    def render_single_seq(self, pred_file):

        ## load the gt model
        ## apply the head rotation the gt pose
        ## render the video with the normal render
        ## add the audio
        ## render the video with the paper view render
        ## use that for the comparison

        max_seq_len=576
        if "sadtalker" in pred_file or "SadTalker" in pred_file:
            print("Running SadTalker", pred_file)
            g_fn =lambda x: scipy.ndimage.filters.gaussian_filter1d(x, sigma=1, axis=0)
            with open(pred_file, 'rb') as fin:
                sub_dict = pickle.load(fin,encoding='latin1')
                pred_pose = g_fn(sub_dict["global_pose"][:max_seq_len])
            seq_name = pred_file.split("/")[-1].split(".")[0]
            seq_name_without_ss = seq_name.split("_ns")[0]
            sample_id = seq_name.split("_ns")[-1][0]
            new_seq_name = seq_name_without_ss + "_ss%02d" % int(sample_id)
            seq_name = new_seq_name

        elif "talkshow" in pred_file or "TalkShow" in pred_file or "TalkSHOW" in pred_file:
            print("Running Talkshow", pred_file)
            seq_name = pred_file.split("/")[-1].split(".")[0]
            seq_name_without_ss = seq_name.split("_ss")[0]
            ss_id = int(seq_name.split("_ss")[1][:2])
            pose = np.load(pred_file.replace(seq_name, seq_name_without_ss), allow_pickle=True).item()["head"]
            pred_pose = pose[ss_id][:max_seq_len]
            
            # first frame normalization
            pose_matrix = axis_angle_to_matrix(pred_pose)
            norm_pose_pose_matrix = torch.matmul(torch.linalg.inv(pose_matrix[0]), pose_matrix[:])
            pred_pose = matrix_to_axis_angle(norm_pose_pose_matrix)
            pred_pose = pred_pose.numpy()
        
        elif "projects/dataset/HDTF/metrical_tracker_dict" in pred_file:
            #render gt
            seq_name = pred_file.split("/")[-1].split(".")[0]
            seq_name_without_ss = seq_name
            pred_pose = "gt"
        
        elif "metric_eval" in pred_file: # done working
            print("Running our model", pred_file)
            pose = np.load(pred_file, allow_pickle=True)[0] # T x 3
            pred_pose = pose[:max_seq_len]
            seq_name = pred_file.split("/")[-1].split(".")[0]
            seq_name_without_ss = seq_name.split("_ss")[0]
        
        else: 
            print("pred_file", pred_file)
            raise("Enter a valid type")
        
        if type(pred_pose) is not str:
            print("Load pred", pred_pose.shape)

        metrical_tracker_dict = os.path.join(os.environ["HOME"], "projects/dataset/HDTF", "metrical_tracker_dict")
        gt_file = os.path.join(metrical_tracker_dict, seq_name_without_ss+".pkl")
        with open(gt_file, 'rb') as fin:
            sub_dict = pickle.load(fin,encoding='latin1')
            gt_motion = sub_dict["vertice"].reshape(-1, 15069)[:max_seq_len]

            if pred_pose == "gt":
                pred_file = gt_file
                g_fn =lambda x: scipy.ndimage.filters.gaussian_filter1d(x, sigma=2, axis=0)
                pred_pose = sub_dict["global_pose"][:max_seq_len].numpy()
                pred_pose = g_fn(pred_pose)

        print("Load gt", gt_motion.shape)
        audio_path = os.path.join(os.getenv("HOME"), "projects/dataset/HDTF", "wav")
        audio_file = os.path.join(audio_path, seq_name_without_ss+".wav")        

        out_dir = os.path.dirname(pred_file)+"_video"
        os.makedirs(out_dir, exist_ok=True)
        pred_pose = torch.from_numpy(pred_pose)
        seq_len =  min(gt_motion.shape[0], pred_pose.shape[0])

        gt_motion = gt_motion[:seq_len]
        pred_pose = pred_pose[:seq_len]

        curr_pred = self.render.face_model.apply_neck_rotation(gt_motion, pred_pose).detach()


        nf = 600
        curr_pred = curr_pred[:nf]
        out_file, pred_rendered_images = self.render.visualize_meshes(out_dir, seq_name,
                                                                       audio_file, curr_pred,
                                                                       None)
        
        out_file, pred_rendered_images = self.render.visualize_meshes(out_dir, seq_name+"_only_exprs",
                                                                audio_file, gt_motion,
                                                                None)
    
def add_local_arguments(parser):
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--render_type", type=str, default="blender")
    parser.add_argument("--seq", type=str, default=None)

    parser.set_defaults(unseen=False)
    return parser


if __name__ == "__main__":
    parser = ArgumentParser()
    add_local_arguments(parser)
    args=parse_and_load_from_model(parser)

    if args.model_path is None:
        Warning("using debug mode")
        args.model_path = "/is/rg/ncs/projects/bthambiraja/submission_models/head_reg_16_ip_mp0_02_rd02_3D_stdnorm_waud_skip_vel_loss0010/01welcome_edit_keyframe_test_2kf_180frames_inbetween_test_model000100035_editing_test/npy"
        # args.seq = "01welcome_ss00.npy"
        args.seq = "gt_01welcome.npy"

        # args.model_path = "/is/rg/ncs/projects/bthambiraja/SadTalker_hdtf/processed_to_voca_format"
        # args.model_path = "/is/rg/ncs/projects/bthambiraja/TalkShow/hdtf_subj_wise"
        os.environ["HOME"] = os.path.join(os.environ["HOME"], "work")
        # args.model_path = "/home/bthambiraja/work/projects/dataset/HDTF/metrical_tracker_dict"
        # args.seq = "RD_Radio1_000.pkl"
        
    
    if args.seq is None:
        raise("Invalid seq")
        
    tester = head_render(args.render_type)
    tester.render_single_seq(os.path.join(args.model_path, args.seq))