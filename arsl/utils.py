import os

import torch


def get_device():
    
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def save_checkpoint(model, optimizer, epoch, save_to):
    
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    path = os.path.join(save_to, f"cp_{epoch:02d}.pt")
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, cp_path):
    
    checkpoint = torch.load(cp_path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    return checkpoint["epoch"]
