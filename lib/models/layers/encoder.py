import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Optional
from torch import Tensor
from torch.nn import Identity
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath
from lib.models.layers.frozen_bn import FrozenBatchNorm2d
import copy


## Utils
# Mlp
class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        '''
            Args:
                x (torch.Tensor): (B, L, C), input tensor
            Returns:
                torch.Tensor: (B, L, C), output tensor
        '''
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


## Attention
# Self Attention
class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 attn_pos_encoding_only=False):
        super(SelfAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        if attn_pos_encoding_only:
            self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        else:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.k = nn.Linear(dim, dim, bias=qkv_bias)
            self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_pos_encoding_only = attn_pos_encoding_only

    def forward(self, x, q_ape, k_ape, attn_pos):
        '''
            Args:
                x (torch.Tensor): (B, L, C)
                q_ape (torch.Tensor | None): (1 or B, L, C), absolute positional encoding for q
                k_ape (torch.Tensor | None): (1 or B, L, C), absolute positional encoding for k
                attn_pos (torch.Tensor | None): (1 or B, num_heads, L, L), untied positional encoding
            Returns:
                torch.Tensor: (B, L, C)
        '''
        B, N, C = x.shape

        if self.attn_pos_encoding_only:
            assert q_ape is None and k_ape is None
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            q = x + q_ape if q_ape is not None else x
            q = self.q(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

            k = x + k_ape if k_ape is not None else x
            k = self.k(k).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = self.v(x).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = q @ k.transpose(-2, -1)
        attn = attn * self.scale
        if attn_pos is not None:
            attn = attn + attn_pos
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


# Cross Attention
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 attn_pos_encoding_only=False):
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        if attn_pos_encoding_only:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        else:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.k = nn.Linear(dim, dim, bias=qkv_bias)
            self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_pos_encoding_only = attn_pos_encoding_only

    def forward(self, q, kv, q_ape, k_ape, attn_pos):
        '''
            Args:
                q (torch.Tensor): (B, L_q, C)
                kv (torch.Tensor): (B, L_kv, C)
                q_ape (torch.Tensor | None): (1 or B, L_q, C), absolute positional encoding for q
                k_ape (torch.Tensor | None): (1 or B, L_kv, C), absolute positional encoding for k
                attn_pos (torch.Tensor | None): (1 or B, num_heads, L_q, L_kv), untied positional encoding
            Returns:
                torch.Tensor: (B, L_q, C)
        '''
        B, q_N, C = q.shape
        kv_N = kv.shape[1]

        if self.attn_pos_encoding_only:
            assert q_ape is None and k_ape is None
            q = self.q(q).reshape(B, q_N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            kv = self.kv(kv).reshape(B, kv_N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
        else:
            q = q + q_ape if q_ape is not None else q
            q = self.q(q).reshape(B, q_N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = kv + k_ape if k_ape is not None else kv
            k = self.k(k).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = self.v(kv).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = q @ k.transpose(-2, -1)
        attn = attn * self.scale
        if attn_pos is not None:
            attn = attn + attn_pos
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, q_N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    

## Fusion
# FeatureFusionLayer : return z, w after self- and cross-attention about z, x
class FeatureFusion(nn.Module):
    def __init__(self,
                 dim, num_heads, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=nn.Identity(), act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_pos_encoding_only=False):
        super(FeatureFusion, self).__init__()
        self.z_norm1 = norm_layer(dim)
        self.x_norm1 = norm_layer(dim)
        self.z_self_attn = SelfAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, attn_pos_encoding_only)
        self.x_self_attn = SelfAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, attn_pos_encoding_only)

        self.z_norm2_1 = norm_layer(dim)
        self.z_norm2_2 = norm_layer(dim)
        self.x_norm2_1 = norm_layer(dim)
        self.x_norm2_2 = norm_layer(dim)

        self.z_x_cross_attention = CrossAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, attn_pos_encoding_only)
        self.x_z_cross_attention = CrossAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, attn_pos_encoding_only)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.z_norm3 = norm_layer(dim)
        self.x_norm3 = norm_layer(dim)
        print(mlp_ratio)
        self.z_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.x_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.drop_path = drop_path

    def forward(self, z, x, z_self_attn_pos, x_self_attn_pos, z_x_cross_attn_pos, x_z_cross_attn_pos):
        z = z + self.drop_path(self.z_self_attn(self.z_norm1(z), None, None, z_self_attn_pos))
        x = x + self.drop_path(self.x_self_attn(self.x_norm1(x), None, None, x_self_attn_pos))

        z = z + self.drop_path(self.z_x_cross_attention(self.z_norm2_1(z), self.x_norm2_1(x), None, None, z_x_cross_attn_pos))
        x = x + self.drop_path(self.x_z_cross_attention(self.x_norm2_2(x), self.z_norm2_2(z), None, None, x_z_cross_attn_pos))

        z = z + self.drop_path(self.z_mlp(self.z_norm3(z)))
        x = x + self.drop_path(self.x_mlp(self.x_norm3(x)))
        return z, x


# FeatureFusionEncoder : positional encoding + fusion layer
class FeatureFusionEncoder(nn.Module):
    def __init__(self, feature_fusion_layers, z_pos_enc, x_pos_enc,
                 z_rel_pos_index, x_rel_pos_index, z_x_rel_pos_index, x_z_rel_pos_index,
                 z_rel_pos_bias_table, x_rel_pos_bias_table, z_x_rel_pos_bias_table, x_z_rel_pos_bias_table):
        super(FeatureFusionEncoder, self).__init__()
        self.layers = nn.ModuleList(feature_fusion_layers)
        self.z_pos_enc = z_pos_enc
        self.x_pos_enc = x_pos_enc
        self.register_buffer('z_rel_pos_index', z_rel_pos_index, False)
        self.register_buffer('x_rel_pos_index', x_rel_pos_index, False)
        self.register_buffer('z_x_rel_pos_index', z_x_rel_pos_index, False)
        self.register_buffer('x_z_rel_pos_index', x_z_rel_pos_index, False)
        self.z_rel_pos_bias_table = z_rel_pos_bias_table
        self.x_rel_pos_bias_table = x_rel_pos_bias_table
        self.z_x_rel_pos_bias_table = z_x_rel_pos_bias_table
        self.x_z_rel_pos_bias_table = x_z_rel_pos_bias_table

    def forward(self, z, x, z_pos, x_pos):
        '''
            Args:
                z (torch.Tensor): (B, L_z, C), template image feature tokens
                x (torch.Tensor): (B, L_x, C), search image feature tokens
                z_pos (torch.Tensor | None): (1 or B, L_z, C), optional positional encoding for z
                x_pos (torch.Tensor | None): (1 or B, L_x, C), optional positional encoding for x
            Returns:
                Tuple[torch.Tensor, torch.Tensor]:
                    (B, L_z, C): template image feature tokens
                    (B, L_x, C): search image feature tokens
        '''
        # Support untied positional encoding only for simplicity
        assert z_pos is None and x_pos is None

        # untied positional encoding
        z_q_pos, z_k_pos = self.z_pos_enc()
        x_q_pos, x_k_pos = self.x_pos_enc()
        z_self_attn_pos = (z_q_pos @ z_k_pos.transpose(-2, -1)).unsqueeze(0)
        x_self_attn_pos = (x_q_pos @ x_k_pos.transpose(-2, -1)).unsqueeze(0)

        z_x_cross_attn_pos = (z_q_pos @ x_k_pos.transpose(-2, -1)).unsqueeze(0)
        x_z_cross_attn_pos = (x_q_pos @ z_k_pos.transpose(-2, -1)).unsqueeze(0)

        # relative positional encoding
        z_self_attn_pos = z_self_attn_pos + self.z_rel_pos_bias_table(self.z_rel_pos_index)
        x_self_attn_pos = x_self_attn_pos + self.x_rel_pos_bias_table(self.x_rel_pos_index)
        z_x_cross_attn_pos = z_x_cross_attn_pos + self.z_x_rel_pos_bias_table(self.z_x_rel_pos_index)
        x_z_cross_attn_pos = x_z_cross_attn_pos + self.x_z_rel_pos_bias_table(self.x_z_rel_pos_index)

        for layer in self.layers:
            z, x = layer(z, x, z_self_attn_pos, x_self_attn_pos, z_x_cross_attn_pos, x_z_cross_attn_pos)

        return z, x


## Positional
# generate_2d_relative_positional_encoding_index
def generate_2d_relative_positional_encoding_index(z_shape, x_shape):
    '''
        z_shape: (z_h, z_w)
        x_shape: (x_h, x_w)
    '''
    z_2d_index_h, z_2d_index_w = torch.meshgrid(torch.arange(z_shape[0]), torch.arange(z_shape[1]))
    x_2d_index_h, x_2d_index_w = torch.meshgrid(torch.arange(x_shape[0]), torch.arange(x_shape[1]))

    z_2d_index_h = z_2d_index_h.flatten(0)
    z_2d_index_w = z_2d_index_w.flatten(0)
    x_2d_index_h = x_2d_index_h.flatten(0)
    x_2d_index_w = x_2d_index_w.flatten(0)

    diff_h = z_2d_index_h[:, None] - x_2d_index_h[None, :]
    diff_w = z_2d_index_w[:, None] - x_2d_index_w[None, :]

    diff = torch.stack((diff_h, diff_w), dim=-1)
    _, indices = torch.unique(diff.view(-1, 2), return_inverse=True, dim=0)
    return indices.view(z_shape[0] * z_shape[1], x_shape[0] * x_shape[1])


# Learned2DPositionalEncoder
class Learned2DPositionalEncoder(nn.Module):
    def __init__(self, dim, w, h):
        super(Learned2DPositionalEncoder, self).__init__()
        self.w_pos = nn.Parameter(torch.empty(w, dim))
        self.h_pos = nn.Parameter(torch.empty(h, dim))
        trunc_normal_(self.w_pos, std=0.02)
        trunc_normal_(self.h_pos, std=0.02)

    def forward(self):
        w = self.w_pos.shape[0]
        h = self.h_pos.shape[0]
        return (self.w_pos[None, :, :] + self.h_pos[:, None, :]).view(h * w, -1)


# Untied2DPositionalEncoder
class Untied2DPositionalEncoder(nn.Module):
    def __init__(self, dim, num_heads, w, h, scale=None, with_q=True, with_k=True):
        super(Untied2DPositionalEncoder, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.pos = Learned2DPositionalEncoder(dim, w, h)
        self.norm = nn.LayerNorm(dim)
        self.pos_q_linear = None
        self.pos_k_linear = None
        if with_q:
            self.pos_q_linear = nn.Linear(dim, dim)
        if with_k:
            self.pos_k_linear = nn.Linear(dim, dim)

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = scale or head_dim ** -0.5

    def forward(self):
        pos = self.norm(self.pos())
        seq_len = pos.shape[0]
        if self.pos_q_linear is not None and self.pos_k_linear is not None:
            pos_q = self.pos_q_linear(pos).view(seq_len, self.num_heads, -1).transpose(0, 1) * self.scale
            pos_k = self.pos_k_linear(pos).view(seq_len, self.num_heads, -1).transpose(0, 1)
            return pos_q, pos_k
        elif self.pos_q_linear is not None:
            pos_q = self.pos_q_linear(pos).view(seq_len, self.num_heads, -1).transpose(0, 1) * self.scale
            return pos_q
        elif self.pos_k_linear is not None:
            pos_k = self.pos_k_linear(pos).view(seq_len, self.num_heads, -1).transpose(0, 1)
            return pos_k
        else:
            raise RuntimeError


# RelativePosition2DEncoder
class RelativePosition2DEncoder(nn.Module):
    def __init__(self, num_heads, embed_size):
        super(RelativePosition2DEncoder, self).__init__()
        self.relative_position_bias_table = nn.Parameter(torch.empty((num_heads, embed_size)))
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, attn_rpe_index):
        '''
            Args:
                attn_rpe_index (torch.Tensor): (*), any shape containing indices, max(attn_rpe_index) < embed_size
            Returns:
                torch.Tensor: (1, num_heads, *)
        '''
        return self.relative_position_bias_table[:, attn_rpe_index].unsqueeze(0)



#######################
#####build-encoder#####
def build_encoder(encoder_layer, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop, dim, z_size, x_size, drop_path):
    z_shape = [z_size, z_size]
    x_shape = [x_size, x_size]
    encoder_layers = []
    for i in range(encoder_layer):
        encoder_layers.append(
            FeatureFusion(dim, num_heads, mlp_ratio, qkv_bias, drop=drop_rate, attn_drop=attn_drop,
                          drop_path=drop_path.allocate(), attn_pos_encoding_only=True)
        )
    z_abs_encoder = Untied2DPositionalEncoder(dim, num_heads, z_shape[0], z_shape[1])
    x_abs_encoder = Untied2DPositionalEncoder(dim, num_heads, x_shape[0], x_shape[1])

    z_self_attn_rel_pos_index = generate_2d_relative_positional_encoding_index(z_shape, z_shape)
    x_self_attn_rel_pos_index = generate_2d_relative_positional_encoding_index(x_shape, x_shape)

    z_x_cross_attn_rel_pos_index = generate_2d_relative_positional_encoding_index(z_shape, x_shape)
    x_z_cross_attn_rel_pos_index = generate_2d_relative_positional_encoding_index(x_shape, z_shape)

    z_self_attn_rel_pos_bias_table = RelativePosition2DEncoder(num_heads, z_self_attn_rel_pos_index.max() + 1)
    x_self_attn_rel_pos_bias_table = RelativePosition2DEncoder(num_heads, x_self_attn_rel_pos_index.max() + 1)
    z_x_cross_attn_rel_pos_bias_table = RelativePosition2DEncoder(num_heads, z_x_cross_attn_rel_pos_index.max() + 1)
    x_z_cross_attn_rel_pos_bias_table = RelativePosition2DEncoder(num_heads, x_z_cross_attn_rel_pos_index.max() + 1)

    return FeatureFusionEncoder(encoder_layers, z_abs_encoder, x_abs_encoder, z_self_attn_rel_pos_index,
                                x_self_attn_rel_pos_index,
                                z_x_cross_attn_rel_pos_index, x_z_cross_attn_rel_pos_index,
                                z_self_attn_rel_pos_bias_table,
                                x_self_attn_rel_pos_bias_table, z_x_cross_attn_rel_pos_bias_table,
                                x_z_cross_attn_rel_pos_bias_table)