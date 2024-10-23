import torch
import torch.nn as nn

#(1,3,3)
tensor = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=torch.float32)

# shape (batch_size, channels, height, width) (1,1,3,3)
tensor = tensor.unsqueeze(0)


conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2)


output = conv_layer(tensor)

print(output)

for i in conv_layer.parameters():
  print(i)


conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=1)

# shape (batch_size, channels, height, width)
input_tensor = torch.tensor([[1,2,3],[4,5,6],[7,8,9]], dtype=torch.float32)

# shape (batch_size, channels, height, width) (1,1,3,3)
input_tensor = input_tensor.unsqueeze(0)

output_tensor = conv_layer(input_tensor)

print("Input Tensor:\n", input_tensor)
print("Output Tensor:\n", output_tensor)


tensor = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=torch.float32)

#  (batch_size, channels, height, width)
tensor = tensor.unsqueeze(0)


conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2)

# Manually set the kernel weights
with torch.no_grad():
    conv_layer.weight = nn.Parameter(torch.tensor([[[[2, 3], [1, 4]]]], dtype=torch.float32))
    conv_layer.bias = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))  # Set bias to one


output = conv_layer(tensor)

print("Output Tensor:\n", output)

tensor = torch.tensor([[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]], dtype=torch.float32)

# (batch_size, channels, height, width)
tensor = tensor.unsqueeze(0)


conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, padding=1)


with torch.no_grad():
    conv_layer.weight = nn.Parameter(torch.tensor([[[[2, 3], [1, 4]]]], dtype=torch.float32))
    conv_layer.bias = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))  # Set bias to one


output = conv_layer(tensor)

# Print the output
print("Output Tensor:\n", output)

import torch
import torch.nn as nn


input_tensor = torch.tensor([[1, 2, 3, 4],
                             [4, 5, 6, 7],
                             [7, 8, 9, 9],
                             [1, 4, 3, 6]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# padding of 1 and stride of 2
conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=2, padding=1, bias=False)

output_tensor = conv_layer(input_tensor)

print(output_tensor)


import torch
import torch.nn as nn


input_tensor = torch.tensor([[1, 2, 3, 4],
                             [4, 5, 6, 7],
                             [7, 8, 9, 9],
                             [1, 4, 3, 6]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# padding of 1 and stride of 2
conv_layer = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=2, stride=2, padding=1, bias=False)


output_tensor = conv_layer(input_tensor)

print(output_tensor)

import torch
import torch.nn as nn


input_tensor = torch.tensor([[[[1, 2, 3, 4],
                               [4, 5, 6, 7],
                               [7, 8, 9, 9],
                               [1, 4, 3, 6]],
                              [[8, 2, 9, 4],
                               [1, 5, 4, 7],
                               [3, 8, 0, 9],
                               [8, 4, 0, 1]]]], dtype=torch.float32)

# padding of 1 and stride of 2
conv_layer = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=2, stride=2, padding=1, bias=False)

output_tensor = conv_layer(input_tensor)

print(output_tensor)

list(conv_layer.parameters())
