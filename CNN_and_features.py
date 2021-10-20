import torch
import torchvision
import torch.nn as nn
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from CNN_mnist import CNN_vgg, CNN_new, print_num_of_parameters

import matplotlib.pyplot as plt


def find_tensor_shapes(model, img):
    print('Studying VGG Model')
    print('Number of layers in the feature extractor part of the model:')
    print(len(model.features))
    print_num_of_parameters(model)

    print('Lets look at the shape of the tensor across each layer....')
    print('original shape..')
    print(img.shape)
    x = img.unsqueeze(0)  # add batch dimension
    x = x.expand(-1, 3, -1, -1)  # copy the channels across rgb
    for i in range(len(model.features)):
        print(model.features[i])
        x = model.features[i](x)
        print(x.shape)


def print_feature_maps(model, img, layer):
    x = img.unsqueeze(0)  # add batch dimension
    x = x.expand(-1, 3, -1, -1)  # copy the channels across rgb
    feature_maps = model.features[:layer](x)
    a = torchvision.utils.make_grid(feature_maps[0, ::].unsqueeze(1), nrow=8)
    plt.imshow(a[0,::].detach())
    plt.show()


if __name__ == '__main__':
    model_vgg = CNN_vgg(pretrained=True)
    model_new = CNN_new()
    model_new_trained = torch.load('CNN_new.pth')

    test_dataset = torchvision.datasets.MNIST(root='../../data', train=False, transform=transforms.ToTensor())

    img, label = test_dataset.__getitem__(20)

    # show the size of the layers in the fresh CNN model
    find_tensor_shapes(model_new, img)
    
    # you can look at the feature maps
    print_feature_maps(model_new, img, layer=-1)
