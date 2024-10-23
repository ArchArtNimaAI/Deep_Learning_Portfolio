import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

transform = transforms.Compose(
    [transforms.ToTensor(),
     #transforms.Normalize((0.5), (0.5))
    ])

#transform = transforms.ToTensor()

mnist_data = datasets.MNIST(root='./data',
                            train=True,
                            download=True,
                            transform=transform)

data_loader = torch.utils.data.DataLoader(dataset=mnist_data,
                                          batch_size=64,
                                          shuffle=True)

# reduce the size
class Autoencoder_Linear(nn.Module):
    def __init__(self):
        super().__init__()
        # (N, 784) -> (N, 3)
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128), # (N, 784) -> (N, 128)
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 3) # -> N, 3
        )

        # (N, 3) -> (N, 784)
        self.decoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model = Autoencoder_Linear()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=1e-3)

'''
model(x)
model.encoder(x)
model.decoder(x)
'''

"""
c = [1,2,3,4,5]
d = [-1,0,2,4,5]

c = 10
d = 12

error = abs(c-d)   #absolute error
error = (c-d)**2   #square error
#mean absolute error
#mean square error
"""

'''
c = [0.1,0.9,0,0,0]
d = [0.2,0.1,0.6,0.1,0]
categorical cross entropy
'''

num_epochs = 25
outputs = []
for epoch in range(num_epochs):
    for (img, _) in data_loader:
        img = img.reshape(-1, 28*28) #-> for Autoencoder_Linear
        recon = model(img)
        loss = criterion(recon, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
    outputs.append((epoch, img, recon))

for j in range(0, num_epochs, 2):
    plt.figure(figsize=(10, 2))
    plt.gray()
    imgs = outputs[j][1].detach().numpy()
    recon = outputs[j][2].detach().numpy()
    for i, item in enumerate(imgs):
        if i >= 10: break
        plt.subplot(2, 10, i+1)
        item = item.reshape(-1, 28,28) #for Autoencoder_Linear
        # item: 1, 28, 28
        plt.imshow(item[0])

    for i, item in enumerate(recon):
        if i >= 10: break
        plt.subplot(2, 10, 10+i+1) # row_length + i + 1
        item = item.reshape(-1, 28,28) #for Autoencoder_Linear
        # item: 1, 28, 28
        plt.imshow(item[0])

data_loader2 = torch.utils.data.DataLoader(dataset=mnist_data,
                                          batch_size=1000,
                                          shuffle=True)

g = iter(data_loader2)
h = next(g)

X,y = h

for X,y in data_loader2:
  break

X.shape

X_new = X.view(-1,784)

X_new.shape

model.eval()

with torch.no_grad():
  z = model.encoder(X_new)

z.shape

j = z.numpy()

j.shape

import pandas as pd
import numpy as np

a = np.array([[1,2,3,4],[5,6,7,8],[6,3,8,9]])

a

df = pd.DataFrame(a)
df

df = pd.DataFrame(j)
df

k = y.numpy()
df2 = pd.DataFrame(k)
df2

df['label'] = df2[0]

df

df.to_csv('z_space.csv')

 a = [[-20, -14, -30]]
s = torch.Tensor(a)

with torch.no_grad():
  predicted = model.decoder(s).detach().numpy()

for i, item in enumerate(predicted):
    item = item.reshape(-1, 28,28) #for Autoencoder_Linear
    # item: 1, 28, 28
    plt.imshow(item[0])
    plt.show()

t = []
r = 0
c = 0
for i in range(-10,9):
  for j in range(-15,4):
    a = [[i, j, -5]]
    s = torch.Tensor(a)
    with torch.no_grad():
      predicted = model.decoder(s).detach().numpy()
    item = item.reshape(-1, 28,28) #for Autoencoder_Linear
    t.append([r,c,item])
    c += 1
  r += 1
  c = 0

# prompt: plt show all images in the t as grid; t[0] is the row number; t[1] is the column number; t[2] is the image

import matplotlib.pyplot as plt

rows = 20
cols = 20

plt.figure(figsize=(10, 10))

for i in range(rows * cols):
  plt.subplot(rows, cols, i + 1)
  plt.imshow(t[i][2][0])  # Assuming t[i][2] is a list containing the image
  plt.axis('off')

plt.show()
