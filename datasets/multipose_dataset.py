import json
import math
import os

import cv2
import torch
import numpy as np
import torch.utils.data as data
import albumentations as A
from albumentations.pytorch import ToTensorV2

from datasets.coco_hp import COCOHP


class MultiPoseDataset(COCOHP):

    def coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox
    
    def open_image(self, image_path):
        image = cv2.imread(image_path)[:, :, ::-1]
        image = image.astype(np.float32)
        return image

    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)

        anns = list(filter(lambda x: x['category_id'] in self._valid_ids and x['iscrowd'] != 1 and x['area'] > 20, anns))
        num_objs = min(len(anns), self.max_objs)

        if self.transforms is None:
            self.transforms = A.Compose([
                A.Normalize(mean=self.cfg.MODEL.PIXEL_MEAN, std=self.cfg.MODEL.PIXEL_STD),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_classes'], min_area=10),
               keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoints_classes']))

        if self.split != 'test':
            img = self.open_image(img_path)

            bboxes = []
            total_keypoints = []
            total_labels_keypoints = []
            for i in range(num_objs):
                ann = anns[i]
                bbox = self.coco_box_to_bbox(ann['bbox'])
                keypoints = np.array(ann['keypoints'], np.float32).reshape(self.num_joints, 3)
                mask = keypoints[:, 2] > 0
                keypoints = keypoints[mask, :2]
                labels_keypoints = np.array(self.cfg.DATASETS.COCO_KEYPOINTS)[mask]
                
                bboxes.append(bbox)
                if len(keypoints) != 0:
                    total_keypoints += keypoints.tolist()
                    total_labels_keypoints += labels_keypoints.tolist()

            bbox_labels = [0] * num_objs
            
            if self.cfg.INPUT.AUG.HORIZONTAL_FLIP:
                if np.random.random() < self.cfg.INPUT.AUG.HORIZONTAL_FLIP_PROB:
                    horizontal_transform = A.Compose([
                        A.HorizontalFlip(p=1)
                    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_classes']),
                       keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoints_classes']))
                        
                    horizontal_transformed = horizontal_transform(
                        image = img,
                        bboxes = bboxes,
                        bbox_classes = bbox_labels,
                        keypoints = total_keypoints,
                        keypoints_classes = total_labels_keypoints
                    )
                    img = horizontal_transformed['image']
                    bboxes = horizontal_transformed['bboxes']
                    bbox_classes = horizontal_transformed['bbox_classes']
                    total_keypoints = horizontal_transformed['keypoints']
                    keypoints_classes = horizontal_transformed['keypoints_classes']
                    total_labels_keypoints = []
                    for key in horizontal_transformed['keypoints_classes']:
                        if key[:4] == 'left':
                            total_labels_keypoints += ['right' + key[4:]]
                            continue
                        if key[:5] == 'right':
                            total_labels_keypoints += ['left' + key[5:]]
                            continue
                        total_labels_keypoints += [key]

            transformed = self.transforms(
                image = img,
                bboxes = bboxes,
                bbox_classes = bbox_labels,
                keypoints = total_keypoints,
                keypoints_classes = total_labels_keypoints
            )                
            
            bboxes = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
            bbox_classes = torch.as_tensor(transformed['bbox_classes'], dtype=torch.int64)
            keypoints = [torch.as_tensor(x, dtype=torch.float32) for x in transformed['keypoints']]
            
            keypoints_classes = transformed['keypoints_classes']

            target = {}
            target['boxes'] = bboxes
            target['labels'] = bbox_classes
            target['keypoints'] = keypoints
            target['keypoints_classes'] = keypoints_classes

            return transformed['image'], target
        
        else:
            x = self.open_image(img_path)
            
            return self.transforms(image=x)['image']






