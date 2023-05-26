import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

# Параметры обучения
batch_size = 128
learning_rate = 0.1
num_epochs = 100
num_classes = 1000 # Количество классов в ImageNet-mini

# Путь к предварительно обученной модели ResNet34 на CUB-200
model_path = "/content/cubs200-resnet34/checkpoint.pth.tar"

# Путь к датасету ImageNet-mini
data_path = "/content/imagenet-mini/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = models.resnet34(pretrained=False)
model.to(device)

model.fc = nn.Linear(model.fc.in_features, num_classes)
model.fc.to(device)

checkpoint = torch.load(model_path, map_location=device)
state_dict = checkpoint['state_dict']

# Переименование ключей весов
new_state_dict = {}
for key, value in state_dict.items():
  if key.startswith('last_linear'):
    new_key = key.replace('last_linear', 'fc')
    new_state_dict[new_key] = value
  else:
    new_state_dict[key] = value

del new_state_dict['fc.weight']
del new_state_dict['fc.bias']
new_state_dict['fc.weight'] = model.fc.weight
new_state_dict['fc.bias'] = model.fc.bias

model.load_state_dict(new_state_dict)

for param in model.parameters():
  param.requires_grad = False

for param in model.fc.parameters():
  param.requires_grad = True

# Трансформации для нормализации и изменения размера изображений
transform = transforms.Compose([
transforms.Resize(224),
transforms.CenterCrop(224),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Загрузка датасета ImageNet-mini
train_dataset = datasets.ImageFolder(root=data_path + '/train', transform=transform)
val_dataset = datasets.ImageFolder(root=data_path + '/val', transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, momentum = 0.5)

lr_step = 30
lr_gamma = 0.1
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
# Обучение модели
total_step = len(train_loader)
for epoch in range(num_epochs):
  for i, (images, labels) in enumerate(train_loader):
    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()

    nn.utils.clip_grad_norm_(model.parameters(), max_norm = 3.0)

    optimizer.step()

    if (i+1) % 100 == 0:
      print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item()}')

  scheduler.step()

  model.eval()
  with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in val_loader:
      images = images.to(device)
      labels = labels.to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the {len(val_dataset)} validation images: {(100 * correct / total)} %')
