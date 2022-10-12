import torch
from InceptionV3 import InceptionV3

block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
model = InceptionV3([block_idx], pretrained=False)
model.eval()

# input images
img = torch.rand((1, 3, 224, 224)).float()
# img = torch.ones((1, 3, 224, 224)).float()

# forward
feature_maps = model(img)

print(feature_maps.shape)
print(feature_maps.mean(), feature_maps.std(), feature_maps.max(), feature_maps.min())
