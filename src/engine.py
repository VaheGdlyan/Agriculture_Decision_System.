import torch  
from tqdm import tqdm 

def train_one_epoch(model, optimizer, data_loader, device, epoch): 
    model.train() 
    total_loss = 0 

    pbar = tqdm(data_loader, desc = f'Epoch {epoch}') 

    for images, targets, in pbar: 
        # Move data to CPU (or GPU if available)
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets] 

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        pbar.set_postfix(loss=losses.item())

    return total_loss / len(data_loader)