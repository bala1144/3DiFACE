import numpy as np
from numpy.core.numeric import indices
import pyrender
from pyrender import primitive
from pyrender import material
import trimesh
import cv2
from scipy.spatial.transform import Rotation
from pyrender import RenderFlags
import os
from tqdm import tqdm
def get_flame_for_visulization(config = {}):
    if len(config) == 0:
        config["flame_model_path"] = os.path.join(os.getenv('HOME'), "projects/NeuralMotionSynthesis/FLAMEModel",
                                                  "model/generic_model.pkl")
    config["batch_size"] = 600
    config["shape_params"] = 100
    config["expression_params"] = 50
    config["pose_params"] = 0
    config["number_worker"] = 8
    config["use_3D_translation"] = False

    from FLAMEModel.FLAME import FLAME
    flame = FLAME(config)

    return flame



class Facerender:
    def __init__(self, intrinsic=(2035.18464, -2070.36928, 257.55392, 256.546816),
                 img_size=(512, 512)):
        self.image_size = img_size
        self.scene = pyrender.Scene(ambient_light=[.75, .75, .75], bg_color=[0, 0, 0])

        # create camera and light
        self.add_camera(intrinsic)
        light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=2e3)
        self.scene.add(light, pose=np.eye(4))
        self.r = pyrender.OffscreenRenderer(*self.image_size)
        self.mesh_node = None
        self.faces = None

    def add_camera(self, intrinsic):
        (fx, fy, Cx, Cy) = intrinsic
        camera = pyrender.camera.IntrinsicsCamera(fx, fy, Cx, Cy,
                                                  znear=0.05, zfar=10.0, name=None)

        camera_rotation = np.eye(4)
        camera_rotation[:3, :3] = Rotation.from_euler('z', 180, degrees=True).as_matrix() @ Rotation.from_euler('y', 0, degrees=True).as_matrix() @ Rotation.from_euler(
            'x', 0, degrees=True).as_matrix()
        camera_translation = np.eye(4)
        camera_translation[:3, 3] = np.array([0, 0, 1])
        camera_pose = camera_rotation @ camera_translation
        self.scene.add(camera, pose=camera_pose)

    def add_face(self, vertices, faces, pose=np.eye(4)):
        """
        Input :
         vertices : N x 3
         faces: F x 3
        """
        if self.mesh_node is not None:
            self.scene.remove_node(self.mesh_node)
        # # disable processing
        # with Stopwatch('Creating trimesh') as f:
        #     tri_mesh = trimesh.Trimesh(vertices, faces, process=False)
        # # print(tri_mesh)
        # with Stopwatch('Creating pyrender mesh') as f:
        #     mesh = pyrender.Mesh.from_trimesh(tri_mesh)

        # with Stopwatch('Creating mesh add') as f:
        #     self.mesh_node = self.scene.add(mesh, pose=pose)
        
                # disable processing
        tri_mesh = trimesh.Trimesh(vertices, faces)
        mesh = pyrender.Mesh.from_trimesh(tri_mesh)
        self.mesh_node = self.scene.add(mesh, pose=pose)
    

    def add_face_v2(self, vertices, faces, pose=np.eye(4)):
        if self.mesh_node is not None:
            self.scene.remove_node(self.mesh_node)
        primitive = [pyrender.Primitive(
                    positions=vertices.copy(),
                    indices=faces,
                    material = pyrender.MetallicRoughnessMaterial(
                    alphaMode='BLEND',
                    baseColorFactor=[0.3, 0.3, 0.3, 1.0],
                    metallicFactor=0.2,
                    roughnessFactor=0.8),
                mode=pyrender.GLTF.TRIANGLES)
                    ]
        mesh = pyrender.Mesh(primitives=primitive, is_visible=True)
        self.mesh_node = self.scene.add(mesh, pose=pose)


    def add_vertics(self, vertices, point_radius=0.01, vertex_colour=[0.0, 0.0, 1.0]):
        sm = trimesh.creation.uv_sphere(radius=point_radius)
        sm.visual.vertex_colors = vertex_colour
        tfs = np.tile(np.eye(4), (vertices.shape[0], 1, 1))
        tfs[:, :3, 3] = vertices
        vertices_Render = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        self.mesh_node = self.scene.add(vertices_Render, pose=np.eye(4))

    def add_obj(self, obj_path):

        if self.mesh_node is not None:
            self.scene.remove_node(self.mesh_node)
        mesh = trimesh.load(obj_path)
        mesh = pyrender.Mesh.from_trimesh(mesh)
        self.mesh_node = self.scene.add(mesh)

    def render(self):
        flags = RenderFlags.SKIP_CULL_FACES
        color, _ = self.r.render(self.scene, flags=flags)
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        return color

    def live(self):
        pyrender.Viewer(self.scene)


    def render_heat_map(self, vertices, faces, dist, pose=np.eye(4)):
        """
        Input :
         vertices : N x 3
         faces: F x 3
        """
        if self.mesh_node is not None:
            self.scene.remove_node(self.mesh_node)

        # disable processing
        tri_mesh = trimesh.Trimesh(vertices, faces, process=False)
        tri_mesh.visual.vertex_colors = dist

        # pyrender mesh
        mesh = pyrender.Mesh.from_trimesh(tri_mesh)
        self.mesh_node = self.scene.add(mesh, pose=pose)

        # flags = RenderFlags.RGBA | RenderFlags.FLAT
        flags = RenderFlags.SKIP_CULL_FACES |  RenderFlags.RGBA | RenderFlags.FLAT
        color, _ = self.r.render(self.scene, flags=flags)
        heat_map = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        return heat_map

    def get_faces(self):

        if self.faces is None:
            flame = get_flame_for_visulization()
            self.faces = flame.faces
            print("Creating faces for the first time")
        return self.faces
    
    def render_seq(self, vertices,
                   desc="frame",
                   faces=None):
        
        if faces is None:
            faces = self.get_faces()

        images = []
        for i in tqdm(range(vertices.shape[0]), desc=f"rendering {desc}"):
            self.add_face(vertices[i], faces)
            colour = self.render()
            if "frame" != desc:
                cv2.putText(colour, desc, (20, 60), # to change based on need
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            images.append(colour)
        
        return images
    
    def dump_seq_to_obj(self, vertices, out_path, faces=None):
        if faces is None:
            faces = self.get_faces()
        desc = out_path.split("/")[-1]
        os.makedirs(out_path, exist_ok=True)
        for i in tqdm(range(vertices.shape[0]), desc=f"Dump {desc}"):
            outfile = os.path.join(out_path, "%04d.obj"%i)
            trimesh.Trimesh(vertices[i].reshape(5023,3), faces).export(outfile)

    def create_video(self, out_vid_file, out_frames, size=(512, 512), input_audio_file=None, fps=30):
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        # fourcc = cv2.VideoWriter_fourcc(*'X264')
        os.makedirs(os.path.dirname(out_vid_file), exist_ok=True)
        writer = cv2.VideoWriter(out_vid_file, fourcc, fps, size)
        for i, frame in tqdm(enumerate(out_frames), desc="writing video"):
            writer.write(frame)
        writer.release()

        if input_audio_file is not None:
            self.add_audio_to_video(out_vid_file, input_audio_file)
        else:
            # reencode for compression
            out_vid_file_enc = out_vid_file.replace(".mp4", "_enc.mp4")
            os.system(f"ffmpeg -i {out_vid_file} {out_vid_file_enc}")            
            os.system(f"mv {out_vid_file_enc} {out_vid_file}")            



    def add_audio_to_video(self, out_vid_file, input_audio_file):
        # add audio to the videos from the method
        out_vid_file_with_audio = out_vid_file.replace(".mp4", "_audio.mp4")
        audio_add_cmd = "ffmpeg -y -i {0} -i {1} -vcodec h264 -ac 2 -channel_layout stereo -pix_fmt yuv420p -shortest {2}".format(
            out_vid_file, input_audio_file, out_vid_file_with_audio
        )
        print("command fore debugging", audio_add_cmd)
        os.system(audio_add_cmd)
        print()
        # remove the old file withut aaudio
        rm_cmd = "rm {0}".format(
            out_vid_file
        )
        print("command fore debugging", rm_cmd)
        os.system(rm_cmd)
        print()

        print("New file with audio", out_vid_file_with_audio)

