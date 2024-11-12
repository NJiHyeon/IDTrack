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
    """Multilayer perceptrion."""
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
    
    
    
## Query
# TargetQueryDecoderLayer
class TargetQueryDecoderLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=nn.Identity(), act_layer=nn.GELU, norm_layer=nn.LayerNorm, way='ind'):
        super(TargetQueryDecoderLayer, self).__init__()
        self.norm_1_query = norm_layer(dim)
        self.norm_1_coord = norm_layer(dim)
        
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop)
        self.norm_2_query = norm_layer(dim)
        self.norm_2_memory = norm_layer(dim)
        
        self.multihead_attn = nn.MultiheadAttention(dim, num_heads, dropout=drop)
        self.norm_3 = norm_layer(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlpz = Mlp(dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        self.drop_path = drop_path
        self.way = way
        
    def forward(self, query, kv, memoryz, memoryx, 
                query_pos, kv_pos, pos_z, pos_x, identity, identity_search,
                tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None):
        '''
            Args:
                query (torch.Tensor) : (B, num_queries, C)
                memory (torch.Tensor) : (B, L, C)
                query_pos (torch.Tensor) : (1 or B, num_queries, C)
                memory_pos (torch.Tensor) : (1 or B, L, C)
            Returns:
                torch.Tensor : (B, num_queries, C)
        '''
        ## [[[1. coord*coord]]]
        if self.way == 'ind' :
            #print('query', query.shape) #[51, 8, 768]
            #print('pos query', query_pos.shape) #[8, 8, 768]
            # RuntimeError: The size of tensor a (51) must match the size of tensor b (8) at non-singleton dimension 0
            c_v = query
            c_q = c_k = self.norm_1_query(query) + query_pos
        elif self.way == 'seq' :
            c_v = kv.permute(1, 0, 2)
            c_q = self.norm_1_query(query) + query_pos
            c_k = (self.norm_1_coord(kv) + kv_pos).permute(1,0,2) #ide
        else :
            ValueError('way is not ind or seq')
            
        query = query + self.drop_path(self.attn(c_q, c_k, value=c_v, attn_mask=tgt_mask,
                                                 key_padding_mask=tgt_key_padding_mask)[0])
        
        ## [[[2. coord*image]]]
        # query
        q2 = self.norm_2_query(query) + query_pos
        # value
        memory = torch.cat((memoryz, memoryx), dim=1) #둘다? memoryx만?
        memory_in = memory.permute(1, 0, 2)
        # key
        pos = torch.cat((pos_z, pos_x), dim=1)
        ide = torch.cat((identity[:, 0, :].repeat(1, pos_z.shape[1], 1), identity[:, 1, :].repeat(1, pos_x.shape[1], 1)), dim=1)
        k2 = (self.norm_2_memory(memory) + pos + ide).permute(1, 0, 2)
        
        # multihead-attention
        query = query + self.drop_path(self.multihead_attn(query=q2, key=k2, value=memory_in, attn_mask=memory_mask,
                                                           key_padding_mask=memory_key_padding_mask)[0])
        query = query + self.drop_path(self.mlpz(self.norm_3(query)))
        
        return query
        

# TargetQueryDecoderBlock
class TargetQueryDecoderBlock(nn.Module):
    def __init__(self, dim, decoder_layers, num_layer):
        super(TargetQueryDecoderBlock, self).__init__()
        self.layers = nn.ModuleList(decoder_layers)
        self.num_layers = num_layer
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, tgt, kv, z, x, kv_pos, pos_z, pos_x, identity, identity_search, 
                query_pos: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None):
        '''
            Args:
                tgt (torch.Tensor) : (B, num_queries, C)
                tgt_key (torch.Tensor) : (B, num_queries, C)
                tgt_value (torch.Tensor) : (B, num_queries, C)
                z (torch.Tensor): (B, L_z, C)
                x (torch.Tensor): (B, L_x, C)
            Returns:
                torch.Tensor: (B, num_queries, C)
        '''
        output = tgt
        for layer in self.layers:
            output = layer(output, kv, z, x, 
                           query_pos, kv_pos, pos_z, pos_x, identity, identity_search,
                           tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask)
        output = self.norm(output)
        return output



#######################
#####build-decoder#####
# decoder_layer, self.drop_path, in_channel, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop, feat_tz, feat_sz
def build_decoder(decoder_layer, drop_path, dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, z_size, x_size, way):
    z_shape = [z_size, z_size]
    x_shape = [x_size, x_size]
    
    num_layers = decoder_layer
    decoder_layers = []
    for _ in range(num_layers):
        decoder_layers.append(TargetQueryDecoderLayer(dim, num_heads, mlp_ratio, qkv_bias, drop=drop_rate,
                                                      attn_drop=attn_drop_rate, drop_path=drop_path.allocate(),
                                                      way=way))
        drop_path.increase_depth()
    decoder = TargetQueryDecoderBlock(dim, decoder_layers, num_layers)
    return decoder