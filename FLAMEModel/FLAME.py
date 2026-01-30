"""
FLAMEModel Layer: Implementation of the 3D Statistical Face model in PyTorch
It is designed in a way to directly plug in as a decoder layer in a 
Deep learning framework for training and testing
It can also be used for 2D or 3D optimisation applications
Author: Soubhik Sanyal
Copyright (c) 2019, Soubhik Sanyal
All rights reserved.
Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.
You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.
Any use of the computer program without a valid license is prohibited and liable to prosecution.
Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.
All rights reserved.
More information about FLAMEModel is available at http://flame.is.tue.mpg.de.
For questions regarding the PyTorch implementation please contact soubhik.sanyal@tuebingen.mpg.de
"""
# Modified from smplx code [https://github.com/vchoutas/smplx] for FLAMEModel

import numpy as np
import torch
import torch.nn as nn
import pickle
import smplx
from smplx.lbs import lbs, batch_rodrigues, vertices2landmarks, find_dynamic_lmk_idx_and_bcoords
from smplx.utils import Struct, to_tensor, to_np, rot_mat_to_euler
from scipy.spatial.transform import Rotation
from face_animation_model.utils.torch_rotation import *
class dict_to_module(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

class FLAME(nn.Module):
    """
    Given flame parameters this class generates a differentiable FLAMEModel function
    which outputs the a mesh and 3D facial landmarks
    """
    def __init__(self, config):
        super(FLAME, self).__init__()
        self.num_vertices = 5023
        if type(config) is dict:
            config = dict_to_module(**config)

        with open(config.flame_model_path, 'rb') as f:
            self.flame_model = Struct(**pickle.load(f, encoding='latin1'))
        self.NECK_IDX = 1
        self.batch_size = config.batch_size
        self.dtype = torch.float32
        # self.use_face_contour = config.use_face_contour
        self.faces = self.flame_model.f
        self.register_buffer('faces_tensor',
                             to_tensor(to_np(self.faces, dtype=np.int64),
                                       dtype=torch.long))

        # Fixing remaining Shape betas
        # There are total 300 shape parameters to control FLAMEModel; But one can use the first few parameters to express
        # the shape. For example 100 shape parameters are used for RingNet project
        default_shape = torch.zeros([self.batch_size, 300-config.shape_params],
                                            dtype=self.dtype, requires_grad=False)
        self.register_parameter('shape_betas', nn.Parameter(default_shape,
                                                      requires_grad=False))

        default_pose = torch.zeros([self.batch_size, 6-config.pose_params],
                                            dtype=self.dtype, requires_grad=False)
        self.register_parameter('default_pose', nn.Parameter(default_pose,
                                                      requires_grad=False))


        # Fixing remaining expression betas
        # There are total 100 shape expression parameters to control FLAMEModel; But one can use the first few parameters to express
        # the expression. For example 50 expression parameters are used for RingNet project
        default_exp = torch.zeros([self.batch_size, 100 - config.expression_params],
                                    dtype=self.dtype, requires_grad=False)
        self.register_parameter('expression_betas', nn.Parameter(default_exp,
                                                            requires_grad=False))

        # Eyeball and neck rotation
        default_eyball_pose = torch.zeros([self.batch_size, 6],
                                    dtype=self.dtype, requires_grad=False)
        self.register_parameter('eye_pose', nn.Parameter(default_eyball_pose,
                                                            requires_grad=False))

        default_neck_pose = torch.zeros([self.batch_size, 3],
                                    dtype=self.dtype, requires_grad=False)
        self.register_parameter('neck_pose', nn.Parameter(default_neck_pose,
                                                            requires_grad=False))

        # Fixing 3D translation since we use translation in the image plane

        self.use_3D_translation = config.use_3D_translation

        default_transl = torch.zeros([self.batch_size, 3],
                                     dtype=self.dtype, requires_grad=False)
        self.register_parameter(
            'transl',
            nn.Parameter(default_transl, requires_grad=False))

        # The vertices of the template model
        self.register_buffer('v_template',
                             to_tensor(to_np(self.flame_model.v_template),
                                       dtype=self.dtype))

        # The shape components
        shapedirs = self.flame_model.shapedirs
        # The shape components
        self.register_buffer(
            'shapedirs',
            to_tensor(to_np(shapedirs), dtype=self.dtype))
        

        j_regressor = to_tensor(to_np(
            self.flame_model.J_regressor), dtype=self.dtype)
        self.register_buffer('J_regressor', j_regressor)

        # Pose blend shape basis
        num_pose_basis = self.flame_model.posedirs.shape[-1]
        posedirs = np.reshape(self.flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs',
                             to_tensor(to_np(posedirs), dtype=self.dtype))

        # indices of parents for each joints
        parents = to_tensor(to_np(self.flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)

        self.register_buffer('lbs_weights',
                             to_tensor(to_np(self.flame_model.weights), dtype=self.dtype))




    def morph(self, expression_params,  neck_pose=None, jaw_pose=None, eye_pose=None, transl=None, shape_params=None):
        """
            Input:
                shape_params: N X number of shape parameters
                expression_params: N X number of expression parameters
                pose_params: N X number of pose parameters
            return:
                vertices: N X V X 3
                landmarks: N X number of landmarks X 3
        """
        bs = expression_params.shape[0]
        if expression_params.shape[-1] < 100:
            expression_params = torch.cat([expression_params, self.expression_betas.expand(bs,-1)], dim=1)

        if shape_params is None:
            shape_params = (shape_params if shape_params is not None else self.shape_betas.expand(bs,-1))
        elif shape_params.shape[-1] < 300:
            shape_params = torch.cat([shape_params, self.shape_betas.expand(bs,-1)], dim=1)

        betas = torch.cat([shape_params.expand(bs, -1), expression_params], dim=1)

        neck_pose = (neck_pose if neck_pose is not None else self.neck_pose.expand(bs,-1))
        eye_pose = (eye_pose if eye_pose is not None else self.eye_pose.expand(bs,-1))
        jaw_pose = (jaw_pose if jaw_pose is not None else self.default_pose[:, :3].expand(bs,-1))

        full_pose = torch.cat([self.default_pose[:, :3].expand(bs,-1), neck_pose, jaw_pose, eye_pose], dim=1)  # N x 15
        template_vertices = self.v_template.unsqueeze(0).expand(bs, -1, -1)

        transl = (transl if transl is not None else self.transl[:bs])
        vertices, _ = lbs(betas, full_pose, template_vertices,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights)

        if self.use_3D_translation:
            vertices += transl.unsqueeze(dim=1)

        return vertices

    def morph_vis(self, current_bs=1, expression_params=None,  neck_pose=None, jaw_pose=None, eye_pose=None, transl=None, shape_params=None):
        """
            Input:
                shape_params: N X number of shape parameters
                expression_params: N X number of expression parameters
                pose_params: N X number of pose parameters
            return:
                vertices: N X V X 3
                landmarks: N X number of landmarks X 3
        """

        shape_params = (shape_params if shape_params is not None else self.shape_betas[:current_bs])
        expression_params = (expression_params if expression_params is not None else self.expression_betas[:current_bs])


        if expression_params.shape[-1] < 100:
            expression_params = torch.cat([expression_params, self.expression_betas[:current_bs]], dim=1)

        if shape_params.shape[-1] < 300:
            shape_params = torch.cat([shape_params, self.shape_betas[:current_bs]], dim=1)

        betas = torch.cat([shape_params, expression_params], dim=1)

        neck_pose = (neck_pose if neck_pose is not None else self.neck_pose[:current_bs])
        eye_pose = (eye_pose if eye_pose is not None else self.eye_pose[:current_bs])
        jaw_pose = (jaw_pose if jaw_pose is not None else self.default_pose[:current_bs,:3])

        full_pose = torch.cat([self.default_pose[:current_bs, :3], neck_pose, jaw_pose, eye_pose], dim=1)  # N x 15
        template_vertices = self.v_template.unsqueeze(0).repeat(current_bs, 1, 1)

        transl = (transl if transl is not None else self.transl[:current_bs])
        vertices, _ = lbs(betas, full_pose, template_vertices,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights)

        if self.use_3D_translation:
            vertices += transl.unsqueeze(dim=1)

        return vertices

    def convert_torch3d_to_pyrender(self, neck_pose):
        return neck_pose

    from torch import Tensor
    # def lbs(
    #         self,
    #         betas: Tensor,
    #         pose: Tensor,
    #         v_template: Tensor,
    #         shapedirs: Tensor,
    #         posedirs: Tensor,
    #         J_regressor: Tensor,
    #         parents: Tensor,
    #         lbs_weights: Tensor,
    #         pose2rot: bool = True,
    # ):
    #     ''' Performs Linear Blend Skinning with the given shape and pose parameters
    #
    #         Parameters
    #         ----------
    #         betas : torch.tensor BxNB
    #             The tensor of shape parameters
    #         pose : torch.tensor Bx(J + 1) * 3
    #             The pose parameters in axis-angle format
    #         v_template torch.tensor BxVx3
    #             The template mesh that will be deformed
    #         shapedirs : torch.tensor 1xNB
    #             The tensor of PCA shape displacements
    #         posedirs : torch.tensor Px(V * 3)
    #             The pose PCA coefficients
    #         J_regressor : torch.tensor JxV
    #             The regressor array that is used to calculate the joints from
    #             the position of the vertices
    #         parents: torch.tensor J
    #             The array that describes the kinematic tree for the model
    #         lbs_weights: torch.tensor N x V x (J + 1)
    #             The linear blend skinning weights that represent how much the
    #             rotation matrix of each part affects each vertex
    #         pose2rot: bool, optional
    #             Flag on whether to convert the input pose tensor to rotation
    #             matrices. The default value is True. If False, then the pose tensor
    #             should already contain rotation matrices and have a size of
    #             Bx(J + 1)x9
    #         dtype: torch.dtype, optional
    #
    #         Returns
    #         -------
    #         verts: torch.tensor BxVx3
    #             The vertices of the mesh after applying the shape and pose
    #             displacements.
    #         joints: torch.tensor BxJx3
    #             The joints of the model
    #     '''
    #
    #     batch_size = max(betas.shape[0], pose.shape[0])
    #     device, dtype = betas.device, betas.dtype
    #
    #     # Add shape contribution
    #     v_shaped = v_template + smplx.lbs.blend_shapes(betas, shapedirs)
    #
    #     # Get the joints
    #     # NxJx3 array
    #     J = smplx.lbs.vertices2joints(J_regressor, v_shaped)
    #
    #     # 3. Add pose blend shapes
    #     # N x J x 3 x 3
    #     ident = torch.eye(3, dtype=dtype, device=device)
    #     if pose2rot:
    #         rot_mats = batch_rodrigues(pose.view(-1, 3)).view(
    #             [batch_size, -1, 3, 3])
    #
    #         pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
    #         # (N x P) x (P, V * 3) -> N x V x 3
    #         pose_offsets = torch.matmul(
    #             pose_feature, posedirs).view(batch_size, -1, 3)
    #     else:
    #         pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
    #         rot_mats = pose.view(batch_size, -1, 3, 3)
    #
    #         pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
    #                                     posedirs).view(batch_size, -1, 3)
    #
    #     v_posed = pose_offsets + v_shaped
    #     # 4. Get the global joint location
    #     J_transformed, A = smplx.lbs.batch_rigid_transform(rot_mats, J, smplx.lbs.parents, dtype=dtype)
    #
    #     # 5. Do skinning:
    #     # W is N x V x (J + 1)
    #     W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    #     # (N x V x (J + 1)) x (N x (J + 1) x 16)
    #     num_joints = J_regressor.shape[0]
    #     T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
    #         .view(batch_size, -1, 4, 4)
    #
    #     homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
    #                                dtype=dtype, device=device)
    #     v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    #     v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))
    #
    #     verts = v_homo[:, :, :3, 0]
    #
    #     return verts, J_transformed

    def apply_neck_rotation(self, in_vertices, neck_pose, pyrender=False):

        if neck_pose.shape[-1] == 6:
            neck_pose = rotation_6d_to_axis_angle(neck_pose)

        with torch.no_grad():
            bs = in_vertices.shape[0]
            if len(in_vertices.shape) == 2:
                in_vertices = in_vertices.reshape(bs, -1, 3)
            if pyrender:
                neck_pose = self.convert_torch3d_to_pyrender(neck_pose)
            full_pose = torch.cat([
                self.default_pose[:, :3].expand(bs, -1),
                neck_pose,
                self.default_pose[:, :3].expand(bs, -1),
                self.eye_pose.expand(bs, -1)
            ], dim=1)
            # full_pose = torch.cat([self.default_pose[:bs, :3], neck_pose, self.default_pose[:bs, :3], self.eye_pose[:bs, :]], dim=1).shape
            betas = torch.cat([self.shape_betas.expand(bs, -1), self.expression_betas.expand(bs, -1)], dim=1)
            # print(self.shapedirs.shape)
            # print("full pose", full_pose.shape)
            # print("betas", betas.shape)
            # print("fin_verticese", in_vertices.shape)
            vertices, _ = lbs(betas, full_pose, in_vertices,
                                   self.shapedirs, self.posedirs,
                                   self.J_regressor, self.parents,
                               self.lbs_weights)
        return vertices.reshape(bs, -1)