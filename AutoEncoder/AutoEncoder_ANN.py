import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToTensor

tensor_transform = transforms.ToTensor()

tensor_transform



train_dataset = datasets.MNIST(root='./data', train=True, transform=ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=tensor_transform, download=True)

# prompt: merge the train dataset and test dataset as a new dataset

from torch.utils.data import ConcatDataset

merged_dataset = ConcatDataset([train_dataset, test_dataset])


i = 0
for _ in merged_dataset:
  i+=1

i

data = DataLoader(merged_dataset, batch_size=128, shuffle=True)

class AE(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = nn.Sequential(
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16)
    )
    self.decoder = nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 28 * 28),
        nn.Sigmoid()
    )

  def forward(self, x):
    latent_vector = self.encoder(x)
    return self.decoder(latent_vector)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AE().to(device)

device

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

view, reshape, unsqueeze(squeeze)

epochs = 20
loss_history = []
generated_images = []

for epoch in range(epochs):
  for batch, (x, _) in enumerate(data):
    x = x.view(x.size(0), -1)
    x = x.to(device)
    optimizer.zero_grad()
    output = model(x)
    loss = loss_fn(output, x)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    if batch % 100 == 0:
      print(f'Epoch: {epoch+1}/{epochs}, Batch: {batch+1}/{len(data)}, Loss: {loss.item()}')

  generated_images.append([epoch, x, output.cpu().detach().numpy()])
  print(f'Epoch: {epoch+1}/{epochs}, Batch: {batch+1}/{len(data)}, Loss: {loss.item()}')





with torch.no_grad():
  for i,j in zip(x,output):
    plt.imshow(i.view(28,28).to('cpu'))
    plt.show()
    plt.imshow(j.view(28,28).to('cpu'))
    plt.show()
    print('-'*30)

with torch.no_grad():
  for i in generated_images:
    print(i[0])
    plt.imshow(i[1][0].view(28,28).to('cpu'))
    plt.show()
    plt.imshow(i[2][0].reshape(28,28))
    plt.show()

generated_images[0][2][2].reshape(28,28)
