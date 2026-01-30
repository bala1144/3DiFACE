import numpy as np
from numpy.core.numeric import indices
import pyrender
from pyrender import primitive
from pyrender import material
import trimesh
import cv2
from scipy.spatial.transform import Rotation
from pyrender import RenderFlags

class Facerender:
    def __init__(self, img_size=(800, 800),
                 material_config=[.1,.2,.4,.4,.7],
                 camera_pose=None,
                 camera_params=None):
    # def __init__(self, img_size=(800, 800), material_config=[.1,.2,.3,.4,.7]):
        """
        material_config = (R,G,B,,m, rf)
        """
        self.image_size = img_size
        self.scene = pyrender.Scene(ambient_light=[.75, .75, .75], bg_color=[255, 255, 255])

        if camera_params is None:
            self.camera_params = {
                                    'c': np.array([img_size[0] / 2, img_size[1] / 2]),
                                    'f': np.array([4754.97941935 / 2, 4754.97941935 / 2])
                                    }
        else:
            self.camera_params = camera_params

            # 'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
            # 'f': np.array([4754.97941935 / 2, 4754.97941935 / 2])}

        # self.camera_params = {'c': np.array([258.810546875, 244.9597930908203]),
        #                       'f': np.array([4754.97941935 / 2, 4754.97941935 / 2])}

        self.frustum = {'near': 0.01, 'far': 3.0, 'height': 800, 'width': 800}

        # create camera and light
        self.add_camera(camera_pose)
        self.add_light()
        self.r = pyrender.OffscreenRenderer(*self.image_size) # w x h
        self.mesh_node = None

        # pyrender material
        self.primitive_material = pyrender.material.MetallicRoughnessMaterial(
            alphaMode='BLEND',
            baseColorFactor=material_config[:3] +[ 1.0], # current setup add more darkenss to material
            # baseColorFactor=[0.8, 0.8, 0.8, 1.0],
            metallicFactor=material_config[3],
            roughnessFactor=material_config[-1],
        )

    def add_camera(self, camera_pose=None):
        camera = pyrender.camera.IntrinsicsCamera(fx=self.camera_params['f'][0],
                                      fy=self.camera_params['f'][1],
                                      cx=self.camera_params['c'][0],
                                      cy=self.camera_params['c'][1],
                                      znear=self.frustum['near'],
                                      zfar=self.frustum['far'], name=None)

        # camera_rotation = np.eye(4)
        # camera_rotation[:3, :3] = Rotation.from_euler('z', 180, degrees=True).as_matrix() @ Rotation.from_euler('y', 0, degrees=True).as_matrix() @ Rotation.from_euler(
        #     'x', 0, degrees=True).as_matrix()
        # camera_translation = np.eye(4)
        # camera_translation[:3, 3] = np.array([0, 0, 1])
        # camera_pose = camera_rotation @ camera_translation

        if camera_pose is None:
            print("Using camera pose")
            camera_pose = [[1, 0, 0, 0],
                           [0, 1, 0, -0.01],
                           [0, 0, 1, 2],
                           [0, 0, 0, 1]]
            
            # # rotation by 30
            # [  1.0000000,  0.0000000,  0.0000000;
            #     0.0000000,  0.8660254, -0.5000000;
            #     0.0000000,  0.5000000,  0.8660254 ]


            # camera_pose = [ [1, 0, 0, 0],
            #                 [0,  0.98, -0.1, 1],
            #                 [0, 0.1, 0.98, 2],
            #                 [0, 0, 0, 1]]

        self.camera_node = self.scene.add(camera, pose=camera_pose)

    def add_light(self,  intensity=1.5, z_offset=0):
        angle = np.pi / 6.0
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = np.array([0, 0, 1.0 - z_offset])
        pos = camera_pose[:3, 3]
        light_color = np.array([1., 1., 1.])
        # light = pyrender.PointLight(color=light_color, intensity=intensity)
        light = pyrender.DirectionalLight(color=light_color, intensity=intensity)

        light_pose = np.eye(4)
        light_pose[:3, 3] = pos
        self.scene.add(light, pose=light_pose.copy())

        light_pose[:3, 3] = cv2.Rodrigues(np.array([angle, 0, 0]))[0].dot(pos)
        self.scene.add(light, pose=light_pose.copy())

        light_pose[:3, 3] = cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos)
        self.scene.add(light, pose=light_pose.copy())

        light_pose[:3, 3] = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos)
        self.scene.add(light, pose=light_pose.copy())

        light_pose[:3, 3] = cv2.Rodrigues(np.array([0, angle, 0]))[0].dot(pos)
        self.scene.add(light, pose=light_pose.copy())

    def add_face(self, vertices, faces, pose=np.eye(4), material=True):
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
        mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=self.primitive_material, smooth=True)
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

    def create_custom_material_and_render(self, custom_material,vertices,  faces, pose=np.eye(4)):
        if self.mesh_node is not None:
            self.scene.remove_node(self.mesh_node)

        tri_mesh = trimesh.Trimesh(vertices, faces)
        mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=custom_material, smooth=True)
        self.mesh_node = self.scene.add(mesh, pose=pose)

        flags = RenderFlags.SKIP_CULL_FACES
        color, _ = self.r.render(self.scene, flags=flags)

        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        return color


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
    
    def render_w_alpha(self):
        flags = RenderFlags.RGBA | RenderFlags.SKIP_CULL_FACES
        color, _ = self.r.render(self.scene, flags=flags)
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        return color



    def render_w_alpha(self):
        flags = RenderFlags.RGBA | RenderFlags.SKIP_CULL_FACES
        color, _ = self.r.render(self.scene, flags=flags)
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        return color


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

        flags = RenderFlags.RGBA | RenderFlags.FLAT
        color, _ = self.r.render(self.scene, flags=flags)
        heat_map = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        return heat_map


    def render_heat_map_with_mask(self, vertices, faces, dist, mask=None, pose=np.eye(4)):
        """
        Input :
         vertices : N x 3
         faces: F x 3
        """
        if self.mesh_node is not None:
            self.scene.remove_node(self.mesh_node)

        # disable processing
        tri_mesh = trimesh.Trimesh(vertices, faces, process=False)


        # if mask is not None:
        #     # # set the heat map colours only for the mask
        #     new_heatmap_colours = tri_mesh.visual.vertex_colors / 255.0
        #     new_heatmap_colours[mask, 0:3] = dist[mask]
        #     new_mask = np.ones((5023,), dtype=bool)
        #     new_mask[mask] = False
        #     new_heatmap_colours[new_mask] = [.1,.2,.4, 1.0]
        #     tri_mesh.visual.vertex_colors = new_heatmap_colours[:, :3]
        # else:
        #     tri_mesh.visual.vertex_colors = dist
        # mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=True)
        # self.mesh_node = self.scene.add(mesh, pose=pose)

        if mask is not None:
            # # set the heat map colours only for the mask
            new_heatmap_colours = tri_mesh.visual.vertex_colors / 255.0
            new_heatmap_colours[mask, 0:3] = dist[mask]
            new_mask = np.ones((5023,), dtype=bool)
            new_mask[mask] = False
            new_heatmap_colours[new_mask] = [.1,.2,.4, 1.0]
            tri_mesh.visual.vertex_colors = new_heatmap_colours[:, :3]

            # material_config = [.1, .2, .4, .4, .7]
            # # pyrender material
            #
            # primitive_material = trimesh.Trimesh.visual.material.from_colour(new_heatmap_colours)
            # # primitive_material = pyrender.material.MetallicRoughnessMaterial(
            # #     alphaMode='BLEND',
            # #     baseColorFactor=[.1,.2,.4, 1.0],  # current setup add more darkenss to material
            # #     baseColorTexture=new_heatmap_colours,
            # #     metallicFactor=material_config[3],
            # #     roughnessFactor=material_config[-1],
            # # )

        else:
            tri_mesh.visual.vertex_colors = dist

        mesh = pyrender.Mesh.from_trimesh(tri_mesh)
        self.mesh_node = self.scene.add(mesh, pose=pose)

        # flags = RenderFlags.RGBA
        flags = RenderFlags.RGBA | RenderFlags.SKIP_CULL_FACES | RenderFlags.FLAT
        color, _ = self.r.render(self.scene, flags=flags)
        heat_map = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        return heat_map


    def render_normal_map(self, vertices, faces, pose=np.eye(4)):
        """
        Input :
         vertices : N x 3
         faces: F x 3
        """
        if self.mesh_node is not None:
            self.scene.remove_node(self.mesh_node)

        # disable processing
        tri_mesh = trimesh.Trimesh(vertices, faces, process=False)
        temp = tri_mesh.vertex_normals * 1.0
        tri_mesh.visual.vertex_colors = (temp + 1.0) / 2.0

        mesh = pyrender.Mesh.from_trimesh(tri_mesh)
        self.mesh_node = self.scene.add(mesh, pose=pose)

        flags = RenderFlags.RGBA | RenderFlags.FLAT
        color, _ = self.r.render(self.scene, flags=flags)
        heat_map = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        return heat_map


    def live(self):
        pyrender.Viewer(self.scene)

#
# class Facerender_800x1200(Facerender):
#     def __init__(self, img_size=(800, 1200), material_config=[.1,.2,.4,.4,.7]):
#         super(Facerender_800x1200, )