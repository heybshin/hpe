# wget https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth

import torch


checkpoint = torch.load("detr-r50-e632da11.pth", map_location='cpu')
del checkpoint["model"]["class_embed.weight"]
del checkpoint["model"]["class_embed.bias"]
torch.save(checkpoint, "detr-r50_no-class-head.pth")

