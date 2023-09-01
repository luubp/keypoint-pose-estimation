import torch
import torch.nn as nn


class SingleHead(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.inconv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1
        )
        self.silu = nn.SiLU()
        self.outconv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1
        )
        
    def forward(self, x):
        x = self.inconv(x)
        x = self.silu(x)
        x = self.outconv(x)
        return x
    
class CenterNetHead(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.heatmap_head = SingleHead(
            cfg.MODEL.CENTERNET.DECONV_CHANNEL[-1],
            cfg.MODEL.CENTERNET.NUM_CLASSES
        ) 
        self.wh_head = SingleHead(64, 2)
        self.offset_head = SingleHead(64, 2)
        
        #keypoint part
        self.xy_keypoints_head = SingleHead(64, 2 * cfg.MODEL.CENTERNET.NUM_KEYPOINTS)
        self.heatmaps_keypoints_head = SingleHead(64, cfg.MODEL.CENTERNET.NUM_KEYPOINTS)
        self.offsets_keypoints_head = SingleHead(64, 2)
        
    def forward(self, x):
        heatmap = self.heatmap_head(x)
        heatmap = torch.sigmoid(heatmap)
        wh = self.wh_head(x)
        offsets = self.offset_head(x)
        
        #keypoint part
        xy_keypoints = self.xy_keypoints_head(x)
        heatmaps_keypoints = self.heatmaps_keypoints_head(x)
        heatmaps_keypoints = torch.sigmoid(heatmaps_keypoints)
        offsets_keypoints = self.offsets_keypoints_head(x)
        
        predictions = {
            'heatmaps': heatmap,
            'whs': wh,
            'offsets': offsets,
            'xy_keypoints': xy_keypoints,
            'heatmaps_keypoints': heatmaps_keypoints,
            'offsets_keypoints': offsets_keypoints
        }
        return predictions