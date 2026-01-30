import torch
import torch.nn as nn
import math
from face_animation_model.model.wav2vec import Wav2Vec2Model
import os
from collections import defaultdict
from torch.nn.functional import leaky_relu
from torch.nn.modules.activation import MultiheadAttention

class struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

def def_value():
    return None

N_SCALE_FACTOR = 2

def get_inplace_activation(activation):

    if activation == "relu":
        return nn.ReLU(True)
    elif activation == "leakyrelu":
        return nn.LeakyReLU(0.01, True)
    elif activation == "swish":
        return nn.SiLU(True)
    elif activation == "sigmoid":
        return nn.Sigmoid()
    else:
        raise("Error: Invalid activation")

def Normalize(in_channels, norm="batch"):

    if norm == "batch":
        return torch.nn.BatchNorm1d(num_features=in_channels, eps=1e-6, affine=True)
    if norm == "batchfalse":
        return torch.nn.BatchNorm1d(num_features=in_channels, eps=1e-6, affine=False)
    elif norm == "Group":
        return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm =="instance":
        return torch.nn.InstanceNorm1d(num_features=in_channels)
    elif norm == "instancefalse":
        return torch.nn.GroupNorm(num_groups=in_channels, num_channels=in_channels, eps=1e-6, affine=False)
    elif norm == "Layer":
        return torch.nn.GroupNorm(num_groups=1, num_channels=in_channels, eps=1e-6, affine=True)
    else:
        raise("Enter a valid norm")
    
def add_norm(norm:str, in_channels:int, enc_layers:list):

    if norm is not None:
        fn = Normalize(in_channels, norm)
        enc_layers.append(fn)

def add_activation(activation:str, layers:list):
    if activation is not None:
        layers.append(get_inplace_activation(activation))


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps])


# simple DEC multi with MLE with identity at the start

def local_Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

def get_infusion_block_with_norm(block_out, activation, padding_type, norm):
    infuse_block = []
    # 2 * block is for the concat model
    infuse_block.append(nn.Conv1d(2 * block_out, block_out, kernel_size=3, stride=1, padding=1,
                                  padding_mode=padding_type))
    add_activation(activation, infuse_block)
    add_norm(norm, block_out, infuse_block)
    return nn.Sequential(*infuse_block)

################################################################################################
####### Encoder model ################
################################################################################################

class Downsample_without_norm(nn.Module):
    def __init__(self, block_in, block_out, padding_type, activation):
        super().__init__()

        mle_layers = []
        # down sample
        mle_layers.append(nn.Conv1d(block_in, block_out, kernel_size=2, stride=2, padding=0))
        add_activation(activation, mle_layers)
        mle_layers.append(nn.Conv1d(block_out, block_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode=padding_type))
        add_activation(activation, mle_layers)
        self.mle_layer = nn.Sequential(*mle_layers)

    def forward(self, input):
        return self.mle_layer(input)

class simple_Enc_with_MLE_with_w_time_embedding(nn.Module):
    def __init__(self,
                 in_channel,
                 ch,
                 activation="swish",
                 use_time_step=False,
                 ch_multi=[1,1],
                 norm=None,
                 padding_type="reflect",
                 nhead=4,
                 **ignore_args):
        super().__init__()

        self.in_embed = nn.Sequential(
            nn.Conv1d(in_channel, ch, kernel_size=1),
            nn.SiLU(True)
        )
        self.in_infuse = get_infusion_block_with_norm(ch, activation, padding_type, norm)

        #  build the MLE model
        mle_layers = nn.ModuleList()
        infusion_layers = nn.ModuleList()
        for i in range(len(ch_multi) - 1): # 2 -1 = 1 [1, 2]
            block_in = ch * ch_multi[i] # 128
            block_out = ch * ch_multi[i + 1] # 128
            mle_layers.append(Downsample_without_norm(block_in, block_out, padding_type, activation)) # 128 -> 128
            infusion_layers.append(get_infusion_block_with_norm(block_out, activation, padding_type, norm))

        self.mle_layers = mle_layers
        self.infusion_layers = infusion_layers

        # create time step embedding
        self.PPE = PositionalEncoding(ch, max_len=5000)
        self.embed_timestep = TimestepEmbedder(ch, self.PPE)

    def forward(self, input, audio, timesteps):
        """
        input: (1x 15069x T)
        audio: (1 x ch_dim x T)
        """
        Bs, audio_dim, T = audio.shape
        x = self.in_embed(input)  # [Bs x 64 x T]

        tim_emb = self.embed_timestep(timesteps)  # [bs, 1, ch]
        tim_emb = tim_emb.permute(0, 2, 1)  # [bs, ch, 1]
        x = x + tim_emb

        x = torch.cat([x,audio], dim=1)  # [128]
        x = self.in_infuse(x)  #

        for down, infuse in zip(self.mle_layers, self.infusion_layers):
            # for down in self.mle_layers:
            x = down(x)
            audio = audio.reshape(Bs, -1, T//2)
            # x = x + audio  # addition based audio condition
            x = torch.cat([x,audio], dim=1) # (Bs, 2*ch, T)
            x = infuse(x)
            T = T//2

        return x

################################################################################################
####### Decoder model ################
################################################################################################

class Upsample_without_norm(nn.Module):
    def __init__(self, block_in, block_out, padding_type, activation):
        super().__init__()
        mle_layers = []
        mle_layers.append(nn.ConvTranspose1d(block_in, block_out, kernel_size=2, stride=2, padding=0)) # upsample
        add_activation(activation, mle_layers)
        mle_layers.append(nn.Conv1d(block_out, block_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode=padding_type)) # upsample
        add_activation(activation, mle_layers)
        self.mle_layer = nn.Sequential(*mle_layers)

    def forward(self, input):
        return self.mle_layer(input)

class decoder_2layer_block_w_audio_new_layers(nn.Module):
    def __init__(self,
                 out_channel,
                 ch,
                 activation="swish",
                 use_time_step=False,
                 ch_multi=[1,1],
                 norm=None,
                 padding_type="reflect",
                 num_identity_classes=8,
                 **ignore_args):
        super().__init__()

        reveresd_ch = ch_multi[::-1]
        mle_layers = nn.ModuleList()
        infusion_layers = nn.ModuleList()

        for i in range(len(reveresd_ch) - 1):
            block_in = ch * reveresd_ch[i]
            block_out = ch * reveresd_ch[i + 1]
            mle_layers.append(Upsample_without_norm(block_in, block_out, padding_type, activation))
            infusion_layers.append(get_infusion_block_with_norm(block_out, activation, padding_type, norm))

        self.mle_layers = mle_layers
        self.infusion_layers = infusion_layers

        # INFO: added style embedding for the model
        self.style_map = nn.Linear(num_identity_classes, ch, bias=False)

        # purposely we set this model to be a liner model
        # self.out_attn = AttnBlock(ch)
        self.out_embed = nn.Linear(ch, out_channel)
        nn.init.constant_(self.out_embed.weight, 0)
        nn.init.constant_(self.out_embed.bias, 0)

    def forward(self, x, audio, one_hot):
        """
        x : Bs x feat_dim x T
        one_hot: Bs x num_classes
        """

        Bs, audio_dim, T = x.shape
        for up, infuse in zip(self.mle_layers, self.infusion_layers):
            # for up in self.mle_layers:
            x = up(x)
            audio = audio.reshape(Bs, -1, T*2)
            x = torch.cat([x,audio], dim=1) # (Bs, 2*ch, T)
            x = infuse(x)
            T = T*2

        style_embed = self.style_map(one_hot)  # (bs, feature_dim)
        style_embed = style_embed.unsqueeze(-1)  # (bs, feature_dim, 1)
        x = x + style_embed
        x = self.out_embed(x.permute(0, 2, 1)).permute(0, 2, 1)  # (Bs x T x 15069)
        return x

################################################################################################
####### Final model ################
################################################################################################

class in_to_motion_unet_w_audio_cond(nn.Module):
    def __init__(self, **args):
        super().__init__()
        """
        audio: (batch_size, raw_wav)
        template: (batch_size, V*3)
        vertice: (batch_size, seq_len, V*3)
        """
        # removed for this experiment
        # # create time step embedding
        # if args["input_mode"] == "pe":
        #     self.in_func = modifided_PositionalEncoding(args["in_channel"], max_len=5000)
        # elif args["input_mode"] == "noise":
        #     from face_animation_model.utils.init_helper import init_from_config
        #     self.in_func = init_from_config(args["noise_cfg"])

        self.encoder = simple_Enc_with_MLE_with_w_time_embedding(**args)
        if args["dec_model"] == "decoder_2layer_block_w_audio_new_layers":
            self.decoder = decoder_2layer_block_w_audio_new_layers(**args)
        elif args["dec_model"] == "decoder_2layer_block_no_audio_new_layers":
            self.decoder = None

        self.cond_mask_prob = args.get('cond_mask_prob', 0.)
        self.cond_mode = args['cond_mode']

        if isinstance(args, dict):
            args = struct(**args)

        # creation of self.args is done in the previous model
        self.create_audio_encoder(args)

        self.args = args
        self.train_subjects = args.train_subjects.split(" ")
        self.dataset = args.dataset
        self.loss = nn.MSELoss()

        # init the model
        if hasattr(args, 'init_from_ckpt') and args.init_from_ckpt is not None:
            ckpt = args.init_from_ckpt
            if ".pt" not in ckpt:
                from face_animation_model.evaluate.eval_root import get_latest_checkpoint
                ckpt = get_latest_checkpoint(os.path.join(ckpt, "checkpoints"))
            print("Init from the checkpoint", ckpt)
            self.init_from_ckpt(path=ckpt)


    def create_audio_encoder(self, args):
        if hasattr(args, 'wav2vec_model'):
            wav2vec_path = os.path.join(os.getenv("HOME"), args.wav2vec_model)
            self.audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec_path)
        else:
            self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

        self.audio_encoder.feature_extractor._freeze_parameters()
        ### added by bala for static feat
        self.audio_encoder.generate_static_audio_features = True
        self.audio_feature_map = nn.Linear(768, args.ch)

    def encode_audio(self, audio, frame_num):
        hidden_states = self.audio_encoder(audio, self.dataset, frame_num=frame_num).last_hidden_state
        hidden_states = self.audio_feature_map(hidden_states)
        return hidden_states

    def forward(self, disp, timesteps, **kwargs):
        """
        disp : B X T X 15069
        timesteps: B X 1
        """
        batch = kwargs["batch"]
        audio, motion, template, one_hot, _ = batch
        self.device = audio.device

        #### INFO : added the audio features
        audio_feat = self.encode_audio(audio, motion.shape[1])  # (Bs, T, ch)

        # add : for the conditional model as unconditional debugging exp
        x = self.encoder(disp.permute(0, 2, 1), audio_feat.permute(0, 2, 1),  timesteps)
        all_disp_out = self.decoder(x, audio_feat.permute(0,2,1), one_hot).permute(0, 2, 1)

        return all_disp_out

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('audio_encoder.')]

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if ik in k:
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]

        unmatched_keys = self.load_state_dict(sd, strict=False)
        print(f"\nRestored from {path}\n")
        print("unmatched keys")
        print(unmatched_keys)

################################################################################################
####### Final model  with masking ################
################################################################################################
class unet_w_cond_mask(in_to_motion_unet_w_audio_cond):
    def __init__(self, **args):
        super().__init__(**args)

    def mask_cond(self, cond, force_mask=False):
        bs, T, d = cond.shape

        if self.training and bs == 1 and self.cond_mask_prob > 0.:
            p = torch.rand(1)
            if p > self.cond_mask_prob:
                return cond
            else:
                return cond * torch.zeros_like(cond)

        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            # print("cond feat", cond.shape)
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1, 1)
            # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            # during inference; we always use the full condition
            return cond

    def forward(self, disp, timesteps, **kwargs):
        """
        disp : B X T X 15069
        timesteps: B X 1
        """
        batch = kwargs["batch"]
        if len(batch) == 5:
            audio, motion, template, one_hot, _ = batch
        else:
            audio, _, template, one_hot, _, motion = batch

        self.device = audio.device

        #### INFO : added the audio features
        audio_feat = self.encode_audio(audio, motion.shape[1])  # (Bs, T, ch)
        audio_feat = self.mask_cond(audio_feat, True if self.cond_mode == "no_cond" else False)
        # add : for the conditional model as unconditional debugging exp
        x = self.encoder(disp.permute(0, 2, 1), audio_feat.permute(0, 2, 1),  timesteps)
        all_disp_out = self.decoder(x, audio_feat.permute(0,2,1), one_hot).permute(0, 2, 1)

        return all_disp_out


################################################################################################
####### Style optim model with masking ################
################################################################################################
class style_optim_w_cond_mask(unet_w_cond_mask):
    def __init__(self, **args):
        super().__init__(**args)
        self.freeze()

    def freeze(self):

        print("Doing style init model")
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("number of trainable params before freezing", trainable_params)

        # freeze the model expect the style emebedding
        for param in self.parameters():
            param.requires_grad = False

        for param in list(self.decoder.style_map.parameters()) + list(self.decoder.out_embed.parameters()):
            param.requires_grad = True

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("number of trainiable params after freezing", trainable_params)

