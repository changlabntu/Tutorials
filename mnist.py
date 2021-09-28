import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

from optparse import OptionParser


def args_train():
    # Training Parameters
    parser = OptionParser()
    # Name of the Project
    parser.add_option('--model', dest='model', default='logistic_regression', type=str, help='type of the model')
    parser.add_option('--mode', type=str, default='dummy')
    parser.add_option('--port', type=str, default='dummy')
    (options, args) = parser.parse_args()
    return options


# define NN architecture
class MLP(nn.Module):
    def __init__(self, hidden_1, hidden_2, dropout=0):
        super(MLP, self).__init__()
        # number of hidden nodes in each layer (512)
        # linear layer (784 -> hidden_1)
        self.fc1 = nn.Linear(28 * 28, hidden_1)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(hidden_2, 10)
        self.droput = nn.Dropout(dropout)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add hidden layer, with relu activation function
        x = F.relu(self.fc2(x))
        # add output layer
        x = self.fc3(x)
        return x


def train(model, args, train_loader, test_loader, loss_function, optimizer):
    """
    model: model you are using
    args: arguments of training
    train_loader: dataloader for training data
    test_loader: dataloader of test data
    loss_function: loss function
    optimizer: optimizer used for the training
    """
    # Train the model
    total_step = len(train_loader)
    for epoch in range(args['num_epochs']):
        # Training
        train_loss = []
        for i, (images, labels) in enumerate(train_loader):
            # Reshape images to (args['batch_size'], img_size)
            images = images.reshape(-1, args['img_size'])

            # Forward pass
            outputs = model(images)
            loss = loss_function(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        # Testing
        test_loss = []
        with torch.no_grad():
            test_step = len(test_loader)
            for i, (images, labels) in enumerate(test_loader):
                # Reshape images to (args['batch_size'], img_size)
                images = images.reshape(-1, args['img_size'])

                # Forward pass
                outputs = model(images)
                loss = loss_function(outputs, labels)

                test_loss.append(loss.item())

        print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}'
              .format(epoch + 1, args['num_epochs'], i + 1, total_step,
                      sum(train_loss) / len(train_loss), sum(test_loss) / len(test_loss)))


# Hyper-parameters
args = {'img_size': 28 * 28,
        'num_classes': 10,
        'num_epochs': 50,
        'batch_size': 16,
        'learning_rate': 0.001}

args.update(vars(args_train()))

# MNIST dataset (images and labels)
train_dataset = torchvision.datasets.MNIST(root='../../data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader (input pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args['batch_size'],
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args['batch_size'],
                                          shuffle=False)


# Logistic regression model
import torch.nn as nn
if args['model'] == 'logistic_regression':
    print('Using logistic regression')
    model = nn.Linear(args['img_size'], args['num_classes'])
elif args['model'] == 'MLP':
    print('MLP')
    model = MLP(dropout=0, hidden_1=512, hidden_2=512)

# Loss and optimizer
# nn.CrossEntropyLoss() computes softmax internally
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args['learning_rate'])

train(model, args, train_loader, test_loader, loss_function, optimizer)

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, args['img_size'])
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
