from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.idtrack.config import cfg, update_config_from_file


def parameters(yaml_name: str):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # update default config from yaml file
    '''multi'''
    # yaml_file = os.path.join(prj_dir, 'experiments/idtrack_aug_pre3_drop50/%s.yaml' % yaml_name) 
    '''each'''
    yaml_file = os.path.join(prj_dir, 'experiments/idtrack_aug_pre5_drop50/%s.yaml' % yaml_name) 
    update_config_from_file(yaml_file)
    params.cfg = cfg
    print("Test config: ", cfg)
    
    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE
    
    # Network checkpoint path
    # multi
    params.checkpoint = os.path.join(save_dir, "checkpoints/train/idtrack/idtrack_aug_pre5_drop50/IDTrack_ep%04d.pth.tar" %
                                     (cfg.TEST.EPOCH))
    # each
    # params.checkpoint = os.path.join(save_dir, "checkpoints/train/idtrack/%s/IDTrack_ep%04d.pth.tar" %
    #                                 (yaml_name, cfg.TEST.EPOCH))
    
    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params