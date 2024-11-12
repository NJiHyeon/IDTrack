class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/mnt/ssd/nozzi/VOT/IDTrack_xyxy_artrack'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/mnt/ssd/nozzi/VOT/IDTrack_xyxy_artrack/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/mnt/ssd/nozzi/VOT/IDTrack_xyxy_artrack/pretrained_networks'
        self.lasot_dir = '/mnt/ssd/nozzi/VOT/IDTrack_xyxy_artrack/data/lasot'
        self.got10k_dir = '/mnt/ssd/nozzi/VOT/IDTrack_xyxy_artrack/data/got10k/train'
        self.got10k_val_dir = '/mnt/ssd/nozzi/VOT/IDTrack_xyxy_artrack/data/got10k/val'
        self.lasot_lmdb_dir = '/mnt/ssd/nozzi/VOT/IDTrack_xyxy_artrack/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/mnt/ssd/nozzi/VOT/IDTrack_xyxy_artrack/data/got10k_lmdb'
        self.trackingnet_dir = '/mnt/ssd/nozzi/VOT/IDTrack_xyxy_artrack/data/trackingnet'
        self.trackingnet_lmdb_dir = '/mnt/ssd/nozzi/VOT/IDTrack_xyxy_artrack/data/trackingnet_lmdb'
        self.coco_dir = '/mnt/ssd/nozzi/VOT/IDTrack_xyxy_artrack/data/coco'
        self.coco_lmdb_dir = '/mnt/ssd/nozzi/VOT/IDTrack_xyxy_artrack/data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/mnt/ssd/nozzi/VOT/IDTrack_xyxy_artrack/data/vid'
        self.imagenet_lmdb_dir = '/mnt/ssd/nozzi/VOT/IDTrack_xyxy_artrack/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''