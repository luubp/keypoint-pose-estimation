import torch
import torch.nn.functional as F


class PredictionDecoder(object):

    @staticmethod
    def decode(pred_dict, cfg):
        
        pred_heatmaps = pred_dict['heatmaps']
        pred_whs = pred_dict['whs']
        pred_offsets = pred_dict['offsets']
        #keypoint part
        pred_xy_keypoints = pred_dict['xy_keypoints']
        pred_heatmaps_keypoints = pred_dict['heatmaps_keypoints']
        pred_offsets_keypoints = pred_dict['offsets_keypoints']
        
        pred_heatmaps = PredictionDecoder.centernet_nms(pred_heatmaps)
        #keypoint part
        #pred_heatmaps_keypoints = centernet_nms(pred_heatmaps_keypoints)
        
        batch_size, num_classes, output_h, output_w = pred_heatmaps.shape
        
        bboxes = []
        cls_scores = []
        cls_labels = []
        
        final_keypoints = []
        final_keypoints_score = []
        
        
        for i in range(batch_size):
            
            heatmap = pred_heatmaps[i].permute(1, 2, 0).view(-1, num_classes)
            wh = pred_whs[i].permute(1, 2, 0).view(-1, 2)
            offset = pred_offsets[i].permute(1, 2, 0).view(-1, 2)
            
            #keypoint part
            xy_keypoint = pred_xy_keypoints[i].permute(1, 2, 0)
            pred_heatmaps_keypoint = pred_heatmaps_keypoints[i]
            pred_offsets_keypoint = pred_offsets_keypoints[i]
            
            yv, xv = torch.meshgrid(torch.arange(0, output_h), torch.arange(0, output_w))
            xv, yv = xv.flatten().float(), yv.flatten().float()
            
            xv, yv = xv.to('cpu'), yv.to('cpu')
            
            
            class_conf, class_pred = torch.max(heatmap, dim=-1)
            #topk
            top_conf, top_ids = torch.topk(class_conf, cfg.MODEL.CENTERNET.MAX_OBJECT_NUM)
            top_true_ids = top_conf >= cfg.MODEL.CENTERNET.BBOX_CONF
            top_ids = top_ids[top_true_ids]
            
            
            class_pred = class_pred[top_ids]
            class_conf = class_conf[top_ids]
            
            #mask = class_conf > confidence
            mask = top_ids
            
            wh_mask = wh[mask]
            offset_mask = offset[mask]
            if len(wh_mask) == 0:
                continue
                
            xv_mask = torch.unsqueeze(xv[mask] + offset_mask[:, 0], -1) #to divide into boxes
            yv_mask = torch.unsqueeze(yv[mask] + offset_mask[:, 1], -1)
            
            half_w, half_h = wh_mask[:, 0:1] / 2, wh_mask[:, 1:2] / 2 #0:1 or 1:2 provides verticality
            
            boxes = torch.cat([xv_mask - half_w, yv_mask - half_h, xv_mask + half_w, yv_mask + half_h], dim=1)
            empty_mask = (boxes[:, 0] != boxes[:, 2]) & (boxes[:, 1] != boxes[:, 3]) #to delete empty boxes
            boxes = boxes[empty_mask][0]
            
            #keypoint part
            center_points = torch.stack([(boxes[0] + boxes[2]) / 2, ((boxes[1] + boxes[3]) / 2)])
            center_points = center_points.type(torch.int64)
            kps = xy_keypoint[center_points[1], center_points[0], :]
            
            bbox_keypoints = []
            bbox_keypoints_scores = []
            bbox_keypoints = []
            for j in range(cfg.MODEL.CENTERNET.NUM_KEYPOINTS):
                kp = kps[2*j:2*j+2]
                if kp[0] == 0 and kp[1] == 0:
                    bbox_keypoints.append([0., 0.])
                    continue
                else:
                    hm_keypoint_x, hm_keypoint_y, kp_score = PredictionDecoder.find_closest_max(pred_heatmaps_keypoint[j], kp, cfg)

                    if hm_keypoint_x == torch.tensor(0) and hm_keypoint_y == torch.tensor(0):
                        bbox_keypoints.append([0, 0])
                        bbox_keypoints_scores.append(0)
                    else:
                        #trick1
                        hm_keypoint_x = (0.9 * hm_keypoint_x + 0.1* kp[0]).type(torch.int64)
                        hm_keypoint_y = (0.9 * hm_keypoint_y + 0.1* kp[1]).type(torch.int64)
                        hm_keypoint_x = torch.clamp(hm_keypoint_x, 0, 127)
                        hm_keypoint_y = torch.clamp(hm_keypoint_y, 0, 127)

                        hm_keypoint_x_off = hm_keypoint_x - pred_offsets_keypoint[0, hm_keypoint_y, hm_keypoint_x]
                        hm_keypoint_y_off = hm_keypoint_y - pred_offsets_keypoint[1, hm_keypoint_y, hm_keypoint_x]
                        bbox_keypoints.append([hm_keypoint_x_off.tolist(), hm_keypoint_y_off.tolist()])
                        bbox_keypoints_scores.append(float(kp_score))
                
            final_keypoints.append(torch.as_tensor(bbox_keypoints))
            final_keypoints_score.append(bbox_keypoints_scores)
            
            #TODO change output type\ add nms
            bboxes.append(boxes)
            cls_scores.append(class_conf[empty_mask][0])
            cls_labels.append(class_pred[empty_mask][0])
        
        if len(bboxes) == 0:
            return dict()
            
        decoded_target = {
        'boxes': torch.stack(bboxes),
        'boxes_scores': torch.stack(cls_scores),
        'boxes_labels': torch.stack(cls_labels),
        'keypoints': torch.stack(final_keypoints),
        'keypoints_score': final_keypoints_score
        }
        
        return decoded_target
    
    @staticmethod
    def find_closest_max(heatmap, point, cfg):
        hm_mask = (heatmap >= cfg.MODEL.CENTERNET.KEYPOINT_HEATMAP_CONF).type(torch.int64)
        h, w = hm_mask.shape
        point = torch.clamp(point, 0, 127).type(torch.int64)
        
        xv, yv = torch.meshgrid(torch.arange(0, w), torch.arange(0, h))
        
        xv_masked, yv_masked = xv - point[1], yv - point[0]
        
        x_mask, y_mask = hm_mask.nonzero(as_tuple=True), hm_mask.nonzero(as_tuple=True)
        
        if x_mask[0].shape[0] == 0:
            return torch.tensor(0), torch.tensor(0), torch.tensor(0)
        
        xv_masked, yv_masked = torch.pow(xv_masked[x_mask], 2), torch.pow(yv_masked[y_mask], 2)
        dist = torch.sqrt(xv_masked + yv_masked)
        
        #add original point if it passed threshold and was not considered
    #     if heatmap[point[1], point[0]] >= cfg.MODEL.CENTERNET.KEYPOINT_HEATMAP_CONF:
    #         print(point[0], point[1], heatmap[point[1], point[0]])
    #         return point[0], point[1], heatmap[point[1], point[0]]
        
        min_idx = dist.argmin()
        keypoint_score = heatmap[yv[y_mask][min_idx], xv[x_mask][min_idx]]
        
        return yv[y_mask][min_idx], xv[x_mask][min_idx], keypoint_score
        
    @staticmethod
    def centernet_nms(heatmap, pool_size=3):
        output_size = heatmap.size()
        pooled_heatmap, inds = F.max_pool2d_with_indices(heatmap, kernel_size=3, return_indices=True)
        nms_heatmap = F.max_unpool2d(pooled_heatmap, indices=inds, kernel_size=3, output_size=output_size)
        return nms_heatmap