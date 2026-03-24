import torchvision 
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor 

def get_wheat_model(num_classes = 2):
    #Loading the model pre-trained on COCO 
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights = 'DEFAULT') 
    
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features 

    # Replace the pre-trained head with a new one (Background + Wheat)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model