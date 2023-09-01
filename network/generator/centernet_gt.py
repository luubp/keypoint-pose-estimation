import torch
from copy import deepcopy
import math
import numpy as np


#DO NOT USE BOX-KEYPOINT-LOSS AUGS!!
class GTGenerator(object):
    
    @staticmethod
    def gt_generation(cfg, target_batch):
        scale = 1 / cfg.MODEL.CENTERNET.DOWN_SCALE
        num_classes = cfg.MODEL.CENTERNET.NUM_CLASSES
        output_size = cfg.INPUT.OUTPUT_SIZE
        min_overlap = cfg.MODEL.CENTERNET.MIN_OVERLAP
        num_keypoints = cfg.MODEL.CENTERNET.NUM_KEYPOINTS
        gt_heatmaps = []
        gt_whs = []
        gt_offsets = []
        gt_reg_masks = []
        
        mapper = dict(zip(cfg.DATASETS.COCO_KEYPOINTS, range(num_keypoints)))
        
        #keypoint part
        gt_xy_keypoints = []
        gt_heatmaps_keypoints = []
        gt_offsets_keypoints = []
        gt_offset_reg_keypoint_masks = []
        gt_xy_reg_masks = []
        
        for target in target_batch:
            
            #create base tensors
            gt_heatmap = torch.zeros(num_classes, output_size[0], output_size[1])
            gt_wh = torch.zeros(2, output_size[0], output_size[1])
            gt_offset = torch.zeros(2, output_size[0], output_size[1])
            gt_reg_mask = torch.zeros(output_size[0], output_size[1])
            
            #keypoint part
            gt_xy_keypoint = torch.zeros(2 * num_keypoints, output_size[0], output_size[1])
            gt_heatmaps_keypoint = torch.zeros(num_keypoints, output_size[0], output_size[1])
            gt_offsets_keypoint = torch.zeros(2, output_size[0], output_size[1])
            gt_xy_reg_mask = torch.zeros(2 * num_keypoints, output_size[0], output_size[1])
            gt_offset_reg_keypoint_mask = torch.zeros(output_size[0], output_size[1])
            
            boxes, classes = target['boxes'].clone(), target['labels'].clone()
            keypoints, keypoint_classes = deepcopy(target['keypoints']), target['keypoints_classes'].copy()
            keypoints_info = target['keypoints_info'].copy()
            
            keypoint_classes = list(map(mapper.get, keypoint_classes))
            
            if len(boxes) != 0:
                boxes = boxes * scale
                keypoints = keypoints * scale
            
                for i in range(len(boxes)):
                    bbox = boxes[i]
                    class_idx = classes[i]
                    h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]

                    gaussian_radius = GTGenerator.get_gaussian_radius(math.ceil(int(h)), math.ceil(int(w)))
                    gaussian_radius = max(0, int(gaussian_radius))

                    center_point = torch.as_tensor([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=torch.float32)
                    center_point_int = center_point.type(torch.int32)

                    gt_heatmap[class_idx, :, :] = GTGenerator.generate_2d_gaussian(gt_heatmap[class_idx, :, :],
                                                                       center_point_int, gaussian_radius)

                    gt_wh[:, center_point_int[1], center_point_int[0]] = torch.as_tensor([w, h], dtype=torch.float32)

                    gt_offset[:, center_point_int[1], center_point_int[0]] = center_point - center_point_int

                    gt_reg_mask[center_point_int[1], center_point_int[0]] = 1

                    #keypoint part
                    check_keypoints, num_joints = keypoints_info[2*i:2*i+2]
                    if check_keypoints:
                        keypoints_labels = keypoint_classes[:num_joints]
                        keypoints_set = keypoints[:num_joints]

                        keypoint_classes = keypoint_classes[num_joints:]
                        keypoints = keypoints[num_joints:]

                        xy_mask = []
                        for idx, j in enumerate(keypoints_labels):
                            xy_mask += [2*j, 2*j + 1]

                            gt_heatmaps_keypoint[j, :, :] = GTGenerator.generate_2d_gaussian(gt_heatmaps_keypoint[j, :, :],
                                                                                            keypoints_set[idx], 5)
                            keypoint_int = keypoints_set[idx].type(torch.int32)
                            gt_offsets_keypoint[:, keypoint_int[1], keypoint_int[0]] = keypoint_int - keypoints_set[idx]
                            gt_offset_reg_keypoint_mask[keypoint_int[1], keypoint_int[0]] = 1

                        gt_xy_keypoint[xy_mask, center_point_int[1], center_point_int[0]] = keypoints_set.flatten()
                        gt_xy_reg_mask = (gt_xy_keypoint > 0).type(torch.int32)     
            
            
                        
            gt_heatmaps.append(gt_heatmap)
            gt_whs.append(gt_wh)
            gt_offsets.append(gt_offset)
            gt_reg_masks.append(gt_reg_mask)     
            #keypoint part
            gt_xy_keypoints.append(gt_xy_keypoint)
            gt_heatmaps_keypoints.append(gt_heatmaps_keypoint)
            gt_offsets_keypoints.append(gt_offsets_keypoint)
            gt_offset_reg_keypoint_masks.append(gt_offset_reg_keypoint_mask)
            gt_xy_reg_masks.append(gt_xy_reg_mask)
             
        gt_dict = {
            "heatmaps": torch.stack(gt_heatmaps),
            "whs": torch.stack(gt_whs),
            "offsets": torch.stack(gt_offsets),
            "reg_masks": torch.stack(gt_reg_masks),
            "xy_keypoints": torch.stack(gt_xy_keypoints),
            "heatmaps_keypoints": torch.stack(gt_heatmaps_keypoints),
            "offsets_keypoints": torch.stack(gt_offsets_keypoints),
            "offset_reg_masks": torch.stack(gt_offset_reg_keypoint_masks),
            "xy_reg_masks": torch.stack(gt_xy_reg_masks)
        }
        
        return gt_dict
    
    @staticmethod
    def get_gaussian_radius(height, width, min_overlap=0.7):

        a1 = 1
        b1 = (height + width)
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2

        return min(r1, r2, r3)

    @staticmethod
    def generate_2d_gaussian(heatmap, center_point, radius):

        diameter = 2 * radius + 1
        gaussian_patch = GTGenerator.gaussian_2d(radius, sigma=diameter / 6)
        gaussian_patch = torch.Tensor(gaussian_patch)

        x, y = int(center_point[0]), int(center_point[1])
        height, width = heatmap.shape

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian_patch = gaussian_patch[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian_patch.shape) > 0 and min(masked_heatmap.shape) > 0:
            masked_heatmap = torch.max(masked_heatmap, masked_gaussian_patch)
            heatmap[y - top:y + bottom, x - left:x + right] = masked_heatmap

        return heatmap

    @staticmethod
    def gaussian_2d(radius, sigma=1):

        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]

        gaussian_patch = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        gaussian_patch[gaussian_patch < np.finfo(gaussian_patch.dtype).eps * gaussian_patch.max()] = 0

        return gaussian_patch