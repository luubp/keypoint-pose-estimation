import torch
import torch.nn.functional as F


def better_focal_loss(pred, gt):
    
    pos_ids = pred.eq(1).float()
    neg_ids = pred.lt(1).float()
    
    neg_weigths = torch.pow(1 - gt, 4)
    #for numerical stability
    pred = torch.clamp(pred, 1e-12)
    
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_ids
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weigths * neg_ids
    
    num_pos = pos_ids.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    
    if num_pos == 0:
        return -neg_loss
    else:
        return -(pos_loss + neg_loss) / num_pos

def l1_loss(pred, gt, mask, xy_mask=None):
    
    expand_mask = mask.unsqueeze(1).repeat(1, 2, 1, 1)
    
    if xy_mask is not None:
        loss = F.smooth_l1_loss(pred*xy_mask, gt*xy_mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss
    else:
        loss = F.smooth_l1_loss(pred * expand_mask, gt * expand_mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss