from google.colab import drive
drive.mount('/content/drive')

import pandas as pd


df = pd.read_csv('from_grasshopper2.csv')

df

df.shape

df = df.sample(frac=1).reset_index(drop=True)
df

import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.feature = torch.tensor(dataframe.iloc[:,:-1].values, dtype=torch.float32)
        # 16(short), 32(int), 64(long)
        self.label = torch.tensor(dataframe.iloc[:,-1].values, dtype=torch.long)
(1000,)
[8,3,2,0,9,8,....,6]
[[0,0,0,0,0,0,0,1,0,0], [0,0,1,0,0,0,0,0,0,0]]
(1000,10)
        # mean , std
        # self.feature_mean = self.feature.mean(dim=0)
        # self.feature_std = self.feature.std(dim=0)
        # self.epsilon = 1e-7

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        X = self.feature[idx]
        y = self.label[idx]
        # normalize
        # X = (X - self.feature_mean) / (self.feature_std + self.epsilon)
        return X,y

from sklearn.model_selection import train_test_split

# train(0.7), test(0.15), val(0.15)
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
test_df, val_df = train_test_split(temp_df, test_size=0.5, random_state=23)

# create datasets for training with pytorch
train_dataset = CustomDataset(train_df)
test_dataset = CustomDataset(test_df)
val_dataset = CustomDataset(val_df)

# create Dataloader
train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=50, shuffle=False)

c = 1
# for X,y in train_loader:
#   print(X) #(5,3)
#   print(y) #(5,)
#   c += 1
#   break

for j in enumerate(train_loader):
  # j = (0,i)
  # i = (X,y)
  # j = (idx,(X,y_real))
  pass

for idx, (X,y_real) in enumerate(train_loader):
  pass

# **Loss Function**

# binary-classification: BCELoss(), multi_class, CrossEntropyLoss():
loss_function = torch.nn.BCELoss()

# **ANN Architecture**

import torch.nn as nn

class ANN(nn.Module):
  def __init__(self, input_size):
    super().__init__()
    self.relu = nn.ReLU()
    self.fc1 = nn.Linear(input_size, 16)
    self.fc2 = nn.Linear(16, 8)
    self.fc3 = nn.Linear(8, 1)
    self.sigmoid = nn.Sigmoid()
    self.bn1 = nn.BatchNorm1d(16)
    self.bn2 = nn.BatchNorm1d(8)
    self.dropout = nn.Dropout(0.2)

  def forward(self,x):
    x = self.fc1(x)
    x = self.relu(x)
    # x = self.bn1(x)
    # x = self.dropout(x)

    x = self.fc2(x)
    x = self.relu(x)
    # x = self.bn2(x)

    x = self.fc3(x)
    y_hat = self.sigmoid(x)
    return y_hat

# **Initialize the model**

model = ANN(3)

# **Optimizer**

import torch.optim as optim

# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# **Training Loops**



```
# epoch
for -----:
  # iterations(mini batch)
  for -----:
```


train_size:700 --- batch_size:50 --- num_batch:14

val_size:150 --- batch_size:50 --- num_batch:3

test_size:150 --- batch_size:50 --- num_batch:3


num_epochs = 100
for epoch in range(num_epochs):
  # training
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    y_pred = model(data)
    loss = loss_function(y_pred, target.unsqueeze(1).float())
    loss.backward()
    optimizer.step()

    if batch_idx % 5 == 0:
      print (f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}')

  # validation
  model.eval()
  with torch.no_grad():
    val_loss = 0.0
    for data, target in val_loader:
      y_pred = model(data)
      loss = loss_function(y_pred, target.unsqueeze(1).float())
      val_loss += loss.item()

    val_loss /= len(val_loader)
    print (f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}')

    # save the model
    if (epoch+1) % 10 == 0:
      torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')

# test
model.eval()
with torch.no_grad():
  test_loss = 0.0
  for data, target in test_loader:
    y_pred = model(data)
    loss = loss_function(y_pred, target.unsqueeze(1).float())

print(test_loss)

model.eval()
with torch.no_grad():
  correct = 0
  total = 0
  for data, target in test_loader:
    y_pred = model(data)
    predicted = (y_pred>0.5).int()
    predicted = predicted.reshape(50)
    correct += (target == predicted).int().sum().item()
    total += len(target)

print(correct)
print(total)
print(correct/total)

# prompt: load prediction dataset for_prediction.csv

import pandas as pd
df_pred = pd.read_csv('for_prediction.csv')
df_pred


# prompt: predict df_pred

# Convert the DataFrame to a PyTorch tensor
features = torch.tensor(df_pred.values, dtype=torch.float32)

# Make predictions
model.eval()
with torch.no_grad():
  predictions = model(features)

# Convert predictions to a list
predicted_classes = (predictions > 0.5).int().squeeze().tolist()

# Print the predictions
print(predicted_classes)


# prompt: write the predicted_classes into a csv file

# Create a DataFrame from the predicted classes
df_predicted = pd.DataFrame({'predicted_classes': predicted_classes})

# Save the DataFrame to a CSV file
df_predicted.to_csv('predictions.csv', index=False)
