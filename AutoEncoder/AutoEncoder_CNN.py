import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, sampler, random_split
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

#Mnist Dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

validation_split = 0.2
train_size = int(len(train_dataset) * (1 - validation_split))
validation_size = len(train_dataset) - train_size

train_dataset, validation_dataset = random_split(train_dataset, [train_size, validation_size])
print(f'Train size: {len(train_dataset)}')
print(f'Validation size: {len(validation_dataset)}')
print(f'Test size: {len(test_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        #Encoder
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        #Decoder
        self.t_conv1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)

    def forward(self, x):
        #Encoder
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        latent = self.pool(x)

        #Decoder
        x = self.relu(self.t_conv1(latent))
        x = torch.sigmoid(self.t_conv2(x))
        return x, latent



model = ConvAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

def train(model, train_loader, criterion, optimizer, device):
  model.train()
  train_loss = 0
  for batch_idx, (data, _) in enumerate(train_loader):
    data = data.to(device)
    optimizer.zero_grad()
    output, _ = model(data)
    loss = criterion(output, data)
    loss.backward()
    optimizer.step()
    train_loss += loss.item()
    if batch_idx % 100 == 0:
      print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
  train_loss /= len(train_loader)
  return train_loss

def validate(model, validation_loader, criterion, device):
  model.eval()
  validation_loss = 0
  with torch.no_grad():
    for data, _ in validation_loader:
      data = data.to(device)
      output, _ = model(data)
      loss = criterion(output, data)
      validation_loss += loss.item()
  validation_loss /= len(validation_loader)
  return validation_loss

#Training Loop
train_losses = []
validation_losses = []

num_epochs = 30
for epoch in range(1, num_epochs+1):
  train_loss = train(model, train_loader, criterion, optimizer, device)
  validation_loss = validate(model, validation_loader, criterion, device)
  train_losses.append(train_loss)
  validation_losses.append(validation_loss)
  print(f'Epoch: {epoch}, Train Loss: {train_loss:.6f}, Validation Loss: {validation_loss:.6f}')

  #plot results
  plt.figure(figsize=(10,5))
  plt.plot(train_losses, label='Train Loss')
  plt.plot(validation_losses, label='Validation Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Training and Validation Loss')
  plt.legend()
  plt.show()

# get a minibatch of test images
dataiter = iter(test_loader)
images, labels = next(dataiter)
images = images.to(device)

#generate image
with torch.no_grad():
  output, latent = model(images)
  output = output.cpu()
  latent = latent.cpu()
  images = images.cpu()
  fig, axs = plt.subplots(6, 10, figsize=(15, 4))
  for i in range(10):
    axs[0, i].imshow(images[i].squeeze(), cmap='gray')
    axs[0, i].axis('off')
    for j in range(1, 5):
      axs[j, i].imshow(latent[i][j].squeeze(), cmap='gray')
      axs[j, i].axis('off')
    axs[5, i].imshow(output[i].squeeze(), cmap='gray')
    axs[5, i].axis('off')

  plt.show()


latent.shape

images.shape
