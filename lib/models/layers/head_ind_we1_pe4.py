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

from lib.models.layers.encoder import build_encoder
from lib.models.layers.decoder import build_decoder

## Utils
# top_k_top_p_filtering_batch
def top_k_top_p_filtering_batch(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:

        for i in range(logits.shape[0]):
            indices_to_remove = logits[i] < torch.topk(logits[i], top_k)[0][..., -1, None]
            logits[i][indices_to_remove] = filter_value

    if top_p > 0.0:
        for i in range(logits.shape[0]):
            sorted_logits, sorted_indices = torch.sort(logits[i], descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[i][indices_to_remove] = filter_value
    return logits


# conv
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
         freeze_bn=False):
    if freeze_bn:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            FrozenBatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
      

# get clones
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# generate_square_mask
def generate_square_subsequent_mask(sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask


# DropPathAllocator
class DropPathAllocator:
    def __init__(self, max_drop_path_rate, stochastic_depth_decay = True):
        self.max_drop_path_rate = max_drop_path_rate
        self.stochastic_depth_decay = stochastic_depth_decay
        self.allocated = []
        self.allocating = []

    def __enter__(self):
        self.allocating = []

    def __exit__(self, exc_type, exc_val, exc_tb):
        if len(self.allocating) != 0:
            self.allocated.append(self.allocating)
        self.allocating = None
        if not self.stochastic_depth_decay:
            for depth_module in self.allocated:
                for module in depth_module:
                    if isinstance(module, DropPath):
                        module.drop_prob = self.max_drop_path_rate
        else:
            depth = self.get_depth()
            dpr = [x.item() for x in torch.linspace(0, self.max_drop_path_rate, depth)]
            assert len(dpr) == len(self.allocated)
            for drop_path_rate, depth_modules in zip(dpr, self.allocated):
                for module in depth_modules:
                    if isinstance(module, DropPath):
                        module.drop_prob = drop_path_rate

    def __len__(self):
        length = 0

        for depth_modules in self.allocated:
            length += len(depth_modules)

        return length

    def increase_depth(self):
        self.allocated.append(self.allocating)
        self.allocating = []

    def get_depth(self):
        return len(self.allocated)

    def allocate(self):
        if self.max_drop_path_rate == 0 or (self.stochastic_depth_decay and self.get_depth() == 0):
            drop_path_module = Identity()
        else:
            drop_path_module = DropPath()
        self.allocating.append(drop_path_module)
        return drop_path_module

    def get_all_allocated(self):
        allocated = []
        for depth_module in self.allocated:
            for module in depth_module:
                allocated.append(module)
        return allocated


## Head Class Option
# Option1 : MLP
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, BN=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        if BN:
            self.layers = nn.ModuleList(nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k))
                                        for n, k in zip([input_dim] + h, h + [output_dim]))
        else:
            self.layers = nn.ModuleList(nn.Linear(n, k)
                                        for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
    
# Option2 : Corner_Predictor
class Corner_Predictor(nn.Module):
    """ Corner Predictor module"""

    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False):
        super(Corner_Predictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        '''top-left corner'''
        self.conv1_tl = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_tl = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_tl = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_tl = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_tl = nn.Conv2d(channel // 8, 1, kernel_size=1)

        '''bottom-right corner'''
        self.conv1_br = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_br = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_br = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_br = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_br = nn.Conv2d(channel // 8, 1, kernel_size=1)

        '''about coordinates and indexs'''
        with torch.no_grad():
            self.indice = torch.arange(0, self.feat_sz).view(-1, 1) * self.stride
            # generate mesh-grid
            self.coord_x = self.indice.repeat((self.feat_sz, 1)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()
            self.coord_y = self.indice.repeat((1, self.feat_sz)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()

    def forward(self, x, return_dist=False, softmax=True):
        """ Forward pass with input x. """
        score_map_tl, score_map_br = self.get_score_map(x)
        if return_dist:
            coorx_tl, coory_tl, prob_vec_tl = self.soft_argmax(score_map_tl, return_dist=True, softmax=softmax)
            coorx_br, coory_br, prob_vec_br = self.soft_argmax(score_map_br, return_dist=True, softmax=softmax)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz, prob_vec_tl, prob_vec_br
        else:
            coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
            coorx_br, coory_br = self.soft_argmax(score_map_br)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz

    def get_score_map(self, x):
        # top-left branch
        x_tl1 = self.conv1_tl(x)
        x_tl2 = self.conv2_tl(x_tl1)
        x_tl3 = self.conv3_tl(x_tl2)
        x_tl4 = self.conv4_tl(x_tl3)
        score_map_tl = self.conv5_tl(x_tl4)

        # bottom-right branch
        x_br1 = self.conv1_br(x)
        x_br2 = self.conv2_br(x_br1)
        x_br3 = self.conv3_br(x_br2)
        x_br4 = self.conv4_br(x_br3)
        score_map_br = self.conv5_br(x_br4)
        return score_map_tl, score_map_br

    def soft_argmax(self, score_map, return_dist=False, softmax=True):
        """ get soft-argmax coordinate for a given heatmap """
        score_vec = score_map.view((-1, self.feat_sz * self.feat_sz))  # (batch, feat_sz * feat_sz)
        prob_vec = nn.functional.softmax(score_vec, dim=1)
        exp_x = torch.sum((self.coord_x * prob_vec), dim=1)
        exp_y = torch.sum((self.coord_y * prob_vec), dim=1)
        if return_dist:
            if softmax:
                return exp_x, exp_y, prob_vec
            else:
                return exp_x, exp_y, score_vec
        else:
            return exp_x, exp_y


# Option3 : CenterPredictor
class CenterPredictor(nn.Module, ):
    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False):
        super(CenterPredictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride

        # corner predict
        self.conv1_ctr = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_ctr = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_ctr = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_ctr = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_ctr = nn.Conv2d(channel // 8, 1, kernel_size=1)

        # size regress
        self.conv1_offset = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_offset = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_offset = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_offset = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_offset = nn.Conv2d(channel // 8, 2, kernel_size=1)

        # size regress
        self.conv1_size = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_size = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_size = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_size = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_size = nn.Conv2d(channel // 8, 2, kernel_size=1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, gt_score_map=None):
        """ Forward pass with input x. """
        score_map_ctr, size_map, offset_map = self.get_score_map(x)

        # assert gt_score_map is None
        if gt_score_map is None:
            bbox = self.cal_bbox(score_map_ctr, size_map, offset_map)
        else:
            bbox = self.cal_bbox(gt_score_map.unsqueeze(1), size_map, offset_map)

        return score_map_ctr, bbox, size_map, offset_map

    def cal_bbox(self, score_map_ctr, size_map, offset_map, return_score=False):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        # bbox = torch.cat([idx_x - size[:, 0] / 2, idx_y - size[:, 1] / 2,
        #                   idx_x + size[:, 0] / 2, idx_y + size[:, 1] / 2], dim=1) / self.feat_sz
        # cx, cy, w, h
        bbox = torch.cat([(idx_x.to(torch.float) + offset[:, :1]) / self.feat_sz,
                          (idx_y.to(torch.float) + offset[:, 1:]) / self.feat_sz,
                          size.squeeze(-1)], dim=1)

        if return_score:
            return bbox, max_score
        return bbox

    def get_pred(self, score_map_ctr, size_map, offset_map):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        # bbox = torch.cat([idx_x - size[:, 0] / 2, idx_y - size[:, 1] / 2,
        #                   idx_x + size[:, 0] / 2, idx_y + size[:, 1] / 2], dim=1) / self.feat_sz
        return size * self.feat_sz, offset

    def get_score_map(self, x):

        def _sigmoid(x):
            y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
            return y

        # ctr branch
        x_ctr1 = self.conv1_ctr(x)
        x_ctr2 = self.conv2_ctr(x_ctr1)
        x_ctr3 = self.conv3_ctr(x_ctr2)
        x_ctr4 = self.conv4_ctr(x_ctr3)
        score_map_ctr = self.conv5_ctr(x_ctr4)

        # offset branch
        x_offset1 = self.conv1_offset(x)
        x_offset2 = self.conv2_offset(x_offset1)
        x_offset3 = self.conv3_offset(x_offset2)
        x_offset4 = self.conv4_offset(x_offset3)
        score_map_offset = self.conv5_offset(x_offset4)

        # size branch
        x_size1 = self.conv1_size(x)
        x_size2 = self.conv2_size(x_size1)
        x_size3 = self.conv3_size(x_size2)
        x_size4 = self.conv4_size(x_size3)
        score_map_size = self.conv5_size(x_size4)
        return _sigmoid(score_map_ctr), _sigmoid(score_map_size), score_map_offset
    
    
# Option4 : Pix2Track
class Pix2Track(nn.Module):
    def __init__(self, in_channel=64, feat_sz=20, feat_tz=10, range=2, pre_num=7, stride=16, bins=400,
                 mlp_ratio=2, qkv_bias=True, drop_rate=0.0, attn_drop=0.0, drop_path=nn.Identity,
                 encoder_layer=3, decoder_layer=3, num_heads=12, way='ind', state='train'):
        super(Pix2Track, self).__init__()
        self.way = way
        self.state = state
        self.bins = bins
        self.range = range
        self.pre_num = pre_num
        self.magic_num = (self.range-1) * 0.5
        
        ## Embedding : word_embedding, pos_embedding
        #1. coordinate word_embedding : q=k=v
        self.word_embeddings = nn.Embedding(self.bins * self.range + 2, in_channel, padding_idx=self.bins * self.range, max_norm=1, norm_type=2.0)
        trunc_normal_(self.word_embeddings.weight, std=.02)
        
        #2. coordinate pos_embedding : q=k=v 
        # test : 아래와 같이 구성 및 idtrack_base 체크포인트 옮기기 (forward도 바꾸기), (yaml 파일 PRETRAIN_PTH 주석처리)
        # new train : 아래와 같이 구성 및 idtrack 체크포인트 옮기기 (forward도 바꾸기)
        self.pos_embeddings = nn.Embedding(5, in_channel)
        self.pos_embeddings_x1 = nn.Embedding(pre_num, in_channel)
        self.pos_embeddings_y1 = nn.Embedding(pre_num, in_channel)
        self.pos_embeddings_x2 = nn.Embedding(pre_num, in_channel)
        self.pos_embeddings_y2 = nn.Embedding(pre_num, in_channel)
        
        #3. bias
        self.output_bias = torch.nn.Parameter(torch.zeros(self.bins * self.range + 2))
        
        self.momentum_param = 0.25
        self.identity_search = torch.nn.Parameter(torch.zeros(1, 1, 768))
        self.identity_search = trunc_normal_(self.identity_search, std=.02)
        self.encoder_layer = encoder_layer
        self.drop_path = drop_path
        self.tz = feat_tz * feat_tz
        self.sz = feat_sz * feat_sz
        
        # Encoder
        if self.encoder_layer > 0 :
            self.encoder = build_encoder(encoder_layer, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop, in_channel, feat_tz, feat_sz, self.drop_path)
        else:
            self.encoder = None
        # Decoder
        self.decoder = build_decoder(decoder_layer, self.drop_path, in_channel, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop, feat_tz, feat_sz, way)


    def forward(self, zx_feat, pos_z, pos_x, identity, seqs_input=None, gt_input=None):
        origin_seq_input = seqs_input
        
        emb_weight = self.word_embeddings.weight.clone()
        share_weight = emb_weight.T

        z_feat = zx_feat[:, :self.tz]
        x_feat = zx_feat[:, self.tz:]

        out_list = []
        bs = zx_feat.shape[0]
        if self.encoder != None :
            z_feat, x_feat = self.encoder(z_feat, x_feat, None, None)
        output_x_feat = x_feat.clone()
        
        ## Past Coordinate make !
        if self.state == 'train':
            n = seqs_input.shape[2]
            seqs_input = seqs_input[..., n - 4*self.pre_num:]
        else :
            seqs_input = seqs_input
        seqs_pre = seqs_input   #[xmin, ymin, xmax, ymax의 과거 좌표들의 나열] shape:[bs, 4*pre_num=28]
        seqs_pre = seqs_pre.view(bs, -1, 4).transpose(1,2).transpose(0,1) #[4, bs=8, pre_num=7]
        seqs_input = seqs_pre.to(zx_feat.device).to(torch.int32) #[4,8,7]

        '''
        seqs_input[i] 
            - shape : [8, 7]
            - config : [[349, 344, 346, 344, 339, 348, 341], ... , [338, 338, 334, 349, 344, 351, 351]]
        word_embed
            - shape : [7, 8, 768]
        decoder_feat_cls
            - shape : [7, 8, 768]
            - 여기서 [8, 7, 768]로 바꿔준 다음 맨 끝의 feature을 선택해서 [8, 768]과 아래의 share_weight가 곱해지고 output_bias가 더해짐
        share_weight 
            - shape : [768, 802]
        output_bias
            - shape : [802]
        out
            - shape : [8, 802]
        extra_seq
            - shape : [8, 1]
            - config : [[707],[707],[707],[707],[507],[707],[ 33],[109]]
        value
            - shape : [8, 1]
            - config : [[0.0128], ... , [0.0060]]
        '''

        for i in range(4) :
            word_embed = self.word_embeddings(seqs_input[i]).permute(1,0,2)
            if i==0 :
                pos_embed = self.pos_embeddings_x1.weight.unsqueeze(1)
            elif i==1 :
                pos_embed = self.pos_embeddings_y1.weight.unsqueeze(1)
            elif i==2 :
                pos_embed = self.pos_embeddings_x2.weight.unsqueeze(1)
            else :
                pos_embed = self.pos_embeddings_y2.weight.unsqueeze(1)
            pos_embed = pos_embed.repeat(1, bs, 1)


            decoder_feat_cls = self.decoder(word_embed, word_embed, z_feat, x_feat, pos_embed, pos_z, pos_x, identity, self.identity_search, pos_embed,
                                            tgt_mask = generate_square_subsequent_mask(len(word_embed)).to(word_embed.device))
            
            out = torch.matmul(decoder_feat_cls.transpose(0,1)[:, -1, :], share_weight) + self.output_bias #근데 왜 여기서 -1???
            out_list.append(out.unsqueeze(0))
            out = out.softmax(-1)
            
            value, extra_seq = out.topk(dim=-1, k=1)[0], out.topk(dim=-1, k=1)[1]
            if i == 0 :
                seqs_output = extra_seq
                values = value
            else :
                seqs_output = torch.cat([seqs_output, extra_seq], dim=-1)
                values = torch.cat([values, value], dim=-1)
            
        if not (not out_list) :
            feat = torch.cat(out_list)

        output = {'seqs': seqs_output, 'class': values, 'feat': feat, 'state': self.state, 'x_teat': output_x_feat.detach()}
        return output
    
        
        
def build_pix_head(cfg, hidden_dim, state):
    way = cfg.MODEL.WAY
    stride = cfg.MODEL.BACKBONE.STRIDE
    in_channel = hidden_dim
    feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
    feat_tz = int(cfg.DATA.TEMPLATE.SIZE / stride)
    decoder_layer = cfg.MODEL.DECODER_LAYER
    encoder_layer = cfg.MODEL.ENCODER_LAYER
    pre_num = cfg.MODEL.PRENUM
    bins = cfg.MODEL.BINS
    range = cfg.MODEL.RANGE
    num_heads = cfg.MODEL.NUM_HEADS
    mlp_ratio = cfg.MODEL.MLP_RATIO
    qkv_bias = cfg.MODEL.QKV_BIAS
    drop_rate = cfg.MODEL.DROP_RATE
    attn_drop = cfg.MODEL.ATTN_DROP
    drop_path = cfg.MODEL.DROP_PATH
    drop_path_allocator = DropPathAllocator(drop_path)
    pix_head = Pix2Track(in_channel=in_channel, feat_sz=feat_sz, feat_tz=feat_tz, range=range, pre_num=pre_num,
                         stride=stride, encoder_layer=encoder_layer, decoder_layer=decoder_layer, bins=bins,
                         num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate,
                         attn_drop=attn_drop, drop_path=drop_path_allocator, way=way, state=state)
    return pix_head

