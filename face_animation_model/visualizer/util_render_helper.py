import os
import numpy as np
from tqdm import tqdm
import cv2
import subprocess
import torch

class render_helper():
    def __init__(self, config = {}, 
                 paper_quality=False, 
                 ratio_800x1200=False, 
                 custom_camera=False,
                 face_model=None,
                 dataset="vocaset"):
        
        if dataset in ["BIWI", "biwi"] :

           #load
            biwi_path = os.path.join("face_animation_model/data/biwi_dataset/BIWI.ply")
            import trimesh 
            face_template = trimesh.load(biwi_path, process=False)
            self.face_model = face_template
            self.fps = 25
            
            intrinsic=(4754.97941935 / 12, -4754.97941935 / 12, 256, 256)


            # camera = pyrender.IntrinsicsCamera(fx=camera_params['f'][0],
            #                         fy=camera_params['f'][1],
            #                         cx=camera_params['c'][0],
            #                         cy=camera_params['c'][1],
            #                         znear=frustum['near'],
            #                         zfar=frustum['far'])


            from face_animation_model.visualizer.util_pyrenderer import Facerender
            self.face_render = Facerender(intrinsic)

        else:
            if face_model is None:

                if dataset == "blend_vocaset":
                    if len(config) == 0:
                        config["flame_model_path"] = os.path.join(os.getenv("HOME"),
                                                      "projects/dataset/BlendVOCA")
                        config["batch_size"] = 1
                    from FLAMEModel.FLAME_Blend import FLAME

                else:
        
                    if len(config) == 0:
                        config["flame_model_path"] = os.path.join(os.getenv('HOME'), "projects/NeuralMotionSynthesis/FLAMEModel",
                                                                "model/generic_model.pkl")
                        config["batch_size"] = 1
                        config["shape_params"] = 0
                        config["expression_params"] = 100
                        config["pose_params"] = 0
                        config["number_worker"] = 8
                        config["use_3D_translation"] = False

                    from FLAMEModel.FLAME import FLAME

                self.face_model = FLAME(config)

            else:
                self.face_model = face_model
                
            self.fps = 30

            from face_animation_model.visualizer.util_pyrenderer import Facerender
            self.face_render = Facerender()


        self.cust_trans = np.zeros((1, 3))
        self.image_size = (512, 512)

    def visualize_meshes_with_gt(self, out_dir, out_seq_name, audio_file, pred_vertices,
                                 gt_vertices):

        out_pred_rendered_images = []
        # for i in tqdm(range(pred_vertices.shape[0]), desc="Rendering pred expressions"):
        for i in range(pred_vertices.shape[0]):
            pred_frame = self.render_exprs_to_image(pred_vertices[i])
            cv2.putText(pred_frame, 'Pred', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            out_pred_rendered_images.append(pred_frame)

        out_gt_rendered_images = []
        # for i in tqdm(range(gt_vertices.shape[0]), desc="Rendering gt expressions"):
        for i in range(gt_vertices.shape[0]):
            gt_frame = self.render_exprs_to_image(gt_vertices[i])
            cv2.putText(gt_frame, 'GT', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(gt_frame, 'Frame no : %03d' % i, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            out_gt_rendered_images.append(gt_frame)

         # create the video out file
        out_vid_file = os.path.join(out_dir, out_seq_name + "_eval_nf%s.mp4" % len(out_pred_rendered_images))
        frame_size = (512 * 2, 512)

        print('Writing video', out_vid_file)
        print()
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        # fourcc = cv2.VideoWriter_fourcc(*'X264')
        writer = cv2.VideoWriter(out_vid_file, fourcc, self.fps, frame_size)
        for i, frame in tqdm(enumerate(out_pred_rendered_images), desc="writing video"):
            out_frame = np.concatenate([out_gt_rendered_images[i], frame], axis=1)
            writer.write(out_frame)
        writer.release()

        self.ffmpeg_encode(out_vid_file)

        # from moviepy.editor import ImageSequenceClip
        # out_gt_rendered_images = np.asarray(out_gt_rendered_images)
        # ImageSequenceClip(out_gt_rendered_images, fps=30).write_videofile(out_vid_file, codec="libx264")

        return  out_vid_file, out_pred_rendered_images

    def ffmpeg_encode(self, out_vid_file):
        new_out_vid_file = out_vid_file.replace(".mp4", "_ff.mp4")
        os.system(f"ffmpeg -y -i {out_vid_file} {new_out_vid_file}")
        os.system(f"rm {out_vid_file}")

    def render_meshes(self, pred_vertices):
        out_pred_rendered_images = []
        for i in tqdm(range(pred_vertices.shape[0]), desc="Rendering expressions"):
            pred_frame = self.render_exprs_to_image(pred_vertices[i])
            out_pred_rendered_images.append(pred_frame)
        return out_pred_rendered_images

    def visualize_meshes(self, out_dir, out_seq_name, audio_file, pred_vertices, annotation=None):

        out_pred_rendered_images = []
        for i in tqdm(range(pred_vertices.shape[0]), desc="Rendering expressions"):
            pred_frame = self.render_exprs_to_image(pred_vertices[i])

            if annotation is not None:
                cv2.putText(pred_frame, annotation, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            out_pred_rendered_images.append(pred_frame)

        # create the video out file
        out_vid_file = os.path.join(out_dir, out_seq_name + "_eval_nf%s.mp4" % len(out_pred_rendered_images))
        video_file = self.compose_write_video(out_vid_file, out_pred_rendered_images, self.image_size, fps=self.fps)

        return  out_vid_file, out_pred_rendered_images

    def visualize_meshes_with_kf(self, out_dir, out_seq_name, pred_vertices, key_frame = [], kf_string="keyframe", desc="in_motion"):

        out_pred_rendered_images = []
        for i in tqdm(range(pred_vertices.shape[0]), desc="Rendering expressions"):
            pred_frame = self.render_exprs_to_image(pred_vertices[i])
            cv2.putText(pred_frame, f"{desc}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                        cv2.LINE_AA)
            
            id = "f: %03d"%i
            colour = (0, 255, 0)
            if i in key_frame or (i-pred_vertices.shape[0]) in key_frame:
                colour = (0, 0, 255)
                id = f"{id} {kf_string}"

            cv2.putText(pred_frame, id, (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2,
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

        # print("Audio file", audio_file)
        # ffmpeg_command = ('ffmpeg -y' + ' -i {0} -i {1} -ac 2 -channel_layout stereo -pix_fmt yuv420p -shortest {2}'.format(
        #     audio_file, video_file, out_vid_w_audio_file))
        # print("ffmpeg_command", ffmpeg_command)
        # out = subprocess.call(ffmpeg_command.split())
        # print("added audio to the video", out_vid_w_audio_file)

        # rm_command = ('rm {0} '.format(video_file))
        # out = subprocess.call(rm_command.split())
        # print("removed ", video_file)

        if os.path.exists(audio_file):
            ffmpeg_command = f"ffmpeg -y -i {video_file} -i {audio_file} -map 0:v -map 1:a -c:v copy -shortest {out_vid_w_audio_file}"
            print("ffmpeg_command", ffmpeg_command)
            os.system(ffmpeg_command)

            rm_command = ('rm {0} '.format(video_file))
            os.system(rm_command)
            print("removed ", video_file)
            print("New file with audio", out_vid_w_audio_file)
        else:
            Warning(f"\naudio file not found {audio_file}")


    def dist_to_rgb(self, errors, min_dist=0.0, max_dist=1.0):
        import matplotlib as mpl
        import matplotlib.cm as cm
        norm = mpl.colors.Normalize(vmin=min_dist, vmax=max_dist)
        cmap = cm.get_cmap(name='jet')
        colormapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        return colormapper.to_rgba(errors)[:, 0:3]

    def error_in_mm(self, pred_verts, gt_verts, vertice_dim=15069):
        pred_verts_mm = pred_verts.view(1,-1, 3) * 1000.0
        gt_verts_mm = gt_verts.view(1,-1, 3) * 1000.0
        diff_in_mm = pred_verts_mm - gt_verts_mm
        dist_in_mm = torch.sqrt(torch.sum(diff_in_mm ** 2, dim=-1))
        return torch.mean(dist_in_mm, dim=0)  # 5023 x 1

    def render_heat_map(self, out_dir, out_seq_name,
                        gt, pred, template, max_error=10):
        out_pred_rendered_images = []
        vid_frames = []
        for i in tqdm(range(pred.shape[0]), desc="Rendering heat map"):
            error = self.error_in_mm(pred[i],
                                     gt[i])
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
