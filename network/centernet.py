import torch
import torchvision
import torch.nn as nn

from .losses import better_focal_loss, l1_loss
from .centernet_deconv import DeconvLayers
from .centernet_head import CenterNetHead
from .generator.centernet_gt import GTGenerator


class CenterNet(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.MODEL.DEVICE
        
        self.backbone = torchvision.models.mobilenet_v3_large(pretrained=True).features
        self.upsample = DeconvLayers(self.cfg)
        self.head = CenterNetHead(self.cfg)
        
    def forward(self, batch):
        
        images, targets = batch
        images = torch.stack([image.to(self.device) for image in images])
        
        features = self.backbone(images)
        up_features = self.upsample(features)
        pred_dict = self.head(up_features)
        
        #gt_dict = GTGenerator.gt_generation(self.cfg, targets)

        #return self.losses(pred_dict, gt_dict)
        return pred_dict
    
    def losses(self, pred_dict, gt_dict):
        
        #heatmap loss
        pred_heatmap = pred_dict['heatmaps']
        current_device = pred_heatmap.device
        
        for gt in gt_dict:
            gt_dict[gt] = gt_dict[gt].to(current_device)
        
        heatmap_loss = better_focal_loss(pred_heatmap, gt_dict['heatmaps'])
        
        #width and height loss
        wh_loss = l1_loss(pred_dict['whs'], gt_dict['whs'], gt_dict['reg_masks'])
        
        #offset loss
        offset_loss = l1_loss(pred_dict['offsets'], gt_dict['offsets'], gt_dict['reg_masks'])
        
        #keypoint heatmap loss
        keypoint_heatmap_loss = better_focal_loss(pred_dict['heatmaps_keypoints'], gt_dict['heatmaps_keypoints'])
        
        #xy keypoint loss
        xy_keypoint_loss = l1_loss(pred_dict['xy_keypoints'], gt_dict['xy_keypoints'], gt_dict['offset_reg_masks'],
                               gt_dict['xy_reg_masks'])
        
        #offset keypoint loss
        offset_keypoint_loss = l1_loss(pred_dict['offsets_keypoints'], gt_dict['offsets_keypoints'],
                                       gt_dict['offset_reg_masks'])
        
        heatmap_loss *= self.cfg.MODEL.LOSS.HEATMAP_WEIGHT
        wh_loss *= self.cfg.MODEL.LOSS.WH_WEIGHT
        offset_loss *= self.cfg.MODEL.LOSS.OFFSET_WEIGHT
        keypoint_heatmap_loss *= self.cfg.MODEL.LOSS.HEATMAP_KEYPOINTS_WEIGHT
        xy_keypoint_loss *= self.cfg.MODEL.LOSS.XY_KEYPOINTS_WEIGHT
        offset_keypoint_loss *= self.cfg.MODEL.LOSS.OFFSET_KEYPOINTS_WEIGHT
        
        loss = {
            'heatmap_loss': heatmap_loss,
            'wh_loss': wh_loss,
            'offset_loss': offset_loss,
            'heatmap_keypoints_loss':  keypoint_heatmap_loss,
            'xy_keypoints_loss': xy_keypoint_loss,
            'offset_keypoints_loss': offset_keypoint_loss
        }
        
        return loss