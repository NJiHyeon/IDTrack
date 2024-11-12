# 예측된 좌표를 추가해서 현재 좌표 예측

import math

from lib.models.idtrack import build_idtrack
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target, transform_image_to_crop
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
import random

    
class IDTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(IDTrack, self).__init__(params)
        network = build_idtrack(params.cfg, training=False)
        print(self.params.checkpoint)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        
        self.cfg = params.cfg
        self.bins = self.cfg.MODEL.BINS 
        self.network = network.cuda() #self.network.eval()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        self.range = self.cfg.MODEL.RANGE 
        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE 
        
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()
        
        # for debug
        self.debug = params.debug 
        self.use_visdom = params.debug 
        self.frame_id = 0
        if self.debug :
            if not self.use_visdom :
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else :
                #self.add_hook()
                self._init_visdom(None, 1)
        
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        self.store_result = None
        self.save_all = 7
        self.x_feat = None
        self.update = None
        self.update_threshold = 5.0
        self.update_intervals = 1
        

    def initialize(self, image, info: dict):
        # Foward the template once
        self.x_feat = None
        
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                                output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        
        with torch.no_grad():
            self.z_dict1 = template
            
        self.box_mask_z = None
        
        # save states
        self.state = info['init_bbox']
        self.store_result = [info['init_bbox'].copy()]
        for i in range(self.save_all - 1) :
            self.store_result.append(info['init_bbox'].copy())
        self.frame_id = 0
        self.update = None
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    # raise ValueError('그만')
    def track(self, image, info: dict = None):
        # print('store_result', self.store_result) [7,4]
        magic_num = (self.range - 1) * 0.5
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size) # (x1, y1, w, h)
        
        #### store_result -> seqs_output make
        for i in range(len(self.store_result)):
            box_temp = self.store_result[i].copy()
            box_out_i = transform_image_to_crop(torch.Tensor(self.store_result[i]), torch.Tensor(self.state),
                                                resize_factor,
                                                torch.Tensor([self.cfg.TEST.SEARCH_SIZE, self.cfg.TEST.SEARCH_SIZE]),
                                                normalize=True)
            box_out_i[2] = box_out_i[2] + box_out_i[0]
            box_out_i[3] = box_out_i[3] + box_out_i[1]
            box_out_i = box_out_i.clamp(min=-0.5, max=1.5)
            box_out_i = (box_out_i + 0.5) * (self.bins - 1)
            if i == 0:
                seqs_out = box_out_i
            else:
                seqs_out = torch.cat((seqs_out, box_out_i), dim=-1)
        #print('store_result', self.store_result) #[7=pre_num,4]
        #print('seqs_out', seqs_out)
        seqs_out = seqs_out.unsqueeze(0)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        
        # network calculate
        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
                #print('seqs_out', seqs_out) : [[28개 과거 좌표]]
            out_dict = self.network.forward(template=self.z_dict1.tensors, search=x_dict.tensors, seq_input=seqs_out)
                #print('out_dict', out_dict)
            
        # add hann windows
            # pred_score_map = out_dict['score_map']
            # response = self.output_window * pred_score_map
            # pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
            # pred_boxes = pred_boxes.view(-1, 4)
         
        # get pred box   
        #print('pred box1', out_dict['seqs'])
        #print('pred box2', out_dict['seqs'][:, 0:4] / (self.bins-1) - magic_num)
        #print('pred box3', out_dict['seqs'][:, 0:4])
        
        pred_boxes = out_dict['seqs'][:, 0:4] / (self.bins-1) - magic_num
        pred_boxes = pred_boxes.view(-1, 4).mean(dim=0)
        pred_new = pred_boxes
        pred_new[2] = pred_boxes[2] - pred_boxes[0] #w
        pred_new[3] = pred_boxes[3] - pred_boxes[1] #h
        pred_new[0] = pred_boxes[0] + pred_new[2]/2 #xmin->centerx
        pred_new[1] = pred_boxes[1] + pred_new[3]/2 #ymin->centery
        pred_boxes = (pred_new * self.params.search_size / resize_factor).tolist()

        # get the final box result
        # 여기 바꿈
        self.state = clip_box(self.map_box_back(pred_boxes, resize_factor), H, W, margin=10)
        
        #### store_result update
        #conf_score = out_dict['class'].sum().item()*10
        #if conf_score > self.update_threshold :
        #    for i in range(self.save_all):
        #        if i != self.save_all - 1 :
        #            self.store_result[i] = self.store_result[i+1]
        #        else:
        #            self.store_result[i] = self.state.copy()
        
        ##원래
        #if len(self.store_result) < self.save_all :
        #    self.store_result.append(self.state.copy())
        #else:
        #    for i in range(self.save_all):
        #        if i != self.save_all - 1 :
        #            self.store_result[i] = self.store_result[i+1]
        #        else:
        #            self.store_result[i] = self.state.copy()
        #print('confi', out_dict['class'].sum().item())
        #print('confi10', out_dict['class'].sum().item()*10)

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        # save_all_boxes
        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}
    

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]


    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []
        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output[1])
            )
            self.enc_attn_weights = enc_attn_weights

    
    
def get_tracker_class():
    return IDTrack