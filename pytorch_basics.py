# Pytorch
import torch
# Pytorch Imaging Toolbox
import torchvision
# Pytorch Neural Network Package
import torch.nn as nn
# Numpy is frequently used
import numpy as np

# New Tensors
x = np.zeros((7, 3, 12, 12))
              # First dim always batch
                 # Second dim always channels
                    # 3rd dim always height
                        # 4th dim always width
                           # (5th can be thickness or temporal...
x = torch.from_numpy(x)
print('New Tensor Shape ' + str(x.shape))
print('New Data Type' + str(x.dtype))

x = torch.rand(7, 3, 12, 12)
print('New Tensor Shape ' + str(x.shape))
print('New Data Type' + str(x.dtype))

# Change Tensor dtype
x = x.type(torch.FloatTensor)
print('New Data Type ' + str(x.dtype))
x = x.type(torch.LongTensor)
print('New Data Type ' + str(x.dtype))

# Change Tensor Shapes
y = x.view(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
print('Changed Tensor Shape ' + str(y.shape))
y = x.view(x.shape[0], -1)
print('Flattened Tensor Shape ' + str(y.shape))

# Concatenate
a = torch.rand(1, 2, 224, 224)
b = torch.rand(1, 1, 224, 224)
c = torch.cat([a, b], dim=1)
print('Concatenated Tensor Shape ' + str(c.shape))

# Slice Tensors
c = c[:, :, 112:, 112:]
print('Sliced Tensor Shape ' + str(c.shape))

# Basic Layers
# Convolutional
conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
# Non-linearity
relu1 = nn.ReLU()  # activation
# Max-pooling
maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1), padding=(0, 0))


# Pretrained Networks
net = torchvision.models.vgg11(pretrained=False)
print(net)
print('number of layers in features:' + str(len(net.features)))
print('number of layers in classifier:' + str(len(net.classifier)))

# Feature
# (224*224 is the default input size for ImageNet and a lot of applications)
input = torch.rand(2, 3, 224, 224)
print('Feature map size:' + str(net.features(input).shape))
print('Output size:' + str(net(input).shape))

# Change the 1000 classes classification to two class
# find the final classification layer
print(net.classifier[-1])
net.classifier[-1] = nn.Linear(in_features=4096, out_features=10)
print(net.classifier[-1])
print('Output size:' + str(net(input).shape))