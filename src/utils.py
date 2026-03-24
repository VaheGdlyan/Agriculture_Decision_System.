import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_img_with_boxes(image, boxes, title="Wheat Detection"):
    """
    image: Tensor of shape (3, H, W)
    boxes: Tensor or list of [x1, y1, x2, y2]
    """
    # Convert tensor (C, H, W) to numpy (H, W, C) for plotting
    img = image.permute(1, 2, 0).cpu().numpy()
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img)
    
    for box in boxes:
        # box is [x1, y1, x2, y2]
        rect = patches.Rectangle(
            (box[0], box[1]), 
            box[2] - box[0], 
            box[3] - box[1], 
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
    
    plt.title(title)
    plt.axis('off')
    plt.show() 

    