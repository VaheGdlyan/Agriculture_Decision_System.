import torch 
from torchvision.ops import box_iou, nms 

def apply_nms(prediction, iou_thresh = 0.5): 
    keep = nms(prediction['boxes'], prediction['scores'], iou_thresh) 
    prediction['boxes'] = prediction['boxes'][keep] 
    prediction['scores'] = prediction['scores'][keep] 
    if 'labels' in prediction: 
        prediction['labels'] = prediction['labels'][keep] 
    return prediction 


def calculate_iou_matrix(preds, targets): 
    if len(preds) == 0 or len(targets) == 0: 
        return torch.zeros((len(preds), len(targets))) 
    return box_iou(preds, targets) 

 
