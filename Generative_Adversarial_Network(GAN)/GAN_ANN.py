import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.optim as optim

#hyperparameters
batch_size = 128
learning_rate = 0.0002
epochs = 200
latent_size = 100
b1 = 0.5
b2 = 0.999

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#data loading
transform = transforms.Compose([transforms.ToTensor(),
                                # change the range from (0,1) ---> (-1,1)
                                transforms.Normalize((0.5,), (0.5,))])

mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
data_loader = DataLoader(mnist, batch_size=batch_size, shuffle=True)

device

#generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128,256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 784),
            nn.Tanh()
        )
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 28, 28)
        return img

#discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid())

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

generator = Generator().to(device)
discriminator = Discriminator().to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(b1, b2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(b1, b2))

loss = nn.BCELoss()

os.makedirs('images', exist_ok=True)

for epoch in range(epochs):
    for i, (imgs, _) in enumerate(data_loader):
        # Adversarial Ground Truths
        valid = torch.ones(imgs.size(0), 1).to(device)
        # valid = [1,1,1,1,1,....,1,1,1]
        fake = torch.zeros(imgs.size(0), 1).to(device)
        # fake = [0,0,0,0,...,0,0,0]

        real_imgs = imgs.to(device)

        # train generator
        optimizer_G.zero_grad()
        z = torch.randn(imgs.size(0), latent_size).to(device)
        gen_imgs = generator(z)
        y_hat = discriminator(gen_imgs)
        g_loss = loss(y_hat, valid)
        g_loss.backward()
        optimizer_G.step()

        # train discriminator
        optimizer_D.zero_grad()
        y_hat_real = discriminator(real_imgs)
        d_real_loss = loss(y_hat_real, valid)
        y_hat_fake = discriminator(gen_imgs.detach())
        d_fake_loss = loss(y_hat_fake, fake)
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        batch_idx = epoch * len(data_loader) + i
        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{epochs}] Batch {i}/{len(data_loader)} \
                      Loss D: {d_loss.item():.4f}, loss G: {g_loss.item():.4f}")

        if batch_idx % 500 == 0:
          save_image(gen_imgs.data[:25], "images/%d.png" % batch_idx, nrow=5, normalize=True)

          # plot
          plt.figure(figsize=(10,5))
          plt.imshow(gen_imgs[0].cpu().detach().numpy().reshape((28,28)), cmap='gray')
          plt.axis('off')
          plt.title(f"Epoch [{epoch}/{epochs}] Batch {i}/{len(data_loader)}")
          plt.show()
          plt.savefig("images/%d.png" % batch_idx)
          plt.close()
