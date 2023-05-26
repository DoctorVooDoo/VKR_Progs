import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os

#Пути к файлам
imagenet_mini_path = "/content/imagenet-mini"
checkpoint_path = "/content/cubs200-resnet34/checkpoint.pth.tar"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Определите преобразования изображений
image_transforms = transforms.Compose([
transforms.Resize((224, 224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

imagenet_mini_dataset = ImageFolder(imagenet_mini_path, transform=image_transforms)
imagenet_mini_loader = DataLoader(imagenet_mini_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

checkpoint = torch.load(checkpoint_path)
state_dict = checkpoint['state_dict']

new_state_dict = {}
for key, value in state_dict.items():
  if key.startswith('last_linear'):
    #замена префикса last_linear на fc для сопоставления
    new_key = 'fc' + key[len('last_linear'):]
    new_state_dict[new_key] = value
  else:
    new_state_dict[key] = value

resnet34_cubs200 = models.resnet34(pretrained=False, num_classes=200)
resnet34_cubs200.load_state_dict(new_state_dict)
resnet34_cubs200 = resnet34_cubs200.to(device)

# классы для хранения извлеченных признаков
features = []
labels = []

with torch.no_grad():
  for images, target_labels in imagenet_mini_loader:
    images = images.to(device)
    target_labels = target_labels.to(device)

    output = resnet34_cubs200(images)
    features.append(output.cpu())
    labels.append(target_labels.squeeze().cpu())

#преобразование списков
features = torch.cat(features)
labels = torch.cat(labels)
