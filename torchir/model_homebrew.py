import os
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

MODEL_LOCATION = 'model/net.model'

SIZE_TRANSFORM = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32)
    ])

PREPROCESSING_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*((0.5,) * 3,) * 2)
    ])

CLASSES = (
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
)

class Net(nn.Module):
    """Your typical cnn for image recognition"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 24, 5)  # convolutional layer
        self.pool = nn.MaxPool2d(2, 2)  # pooling layer
        self.conv2 = nn.Conv2d(24, 16, 5)  # convolutional again
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # fully connected layer
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_network():
    """Train the network using CIFAR10 data"""
    # load the data
    trainset = datasets.CIFAR10(root='./data', train=True, download=True,
                                transform=PREPROCESSING_TRANSFORM)
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    # define network, loss function and optimizer
    net = Net()
    net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # stochastic gradient descent

    # train
    for _ in range(2):
        for data in trainloader:
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # save model
    if MODEL_LOCATION:
        path = os.path.abspath(MODEL_LOCATION)
        print('Saving model to %s' % path)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(net, path)

    return net


def get_network():
    """Get a network - load from disk if available, train othwerise"""
    path = os.path.abspath(MODEL_LOCATION)
    if MODEL_LOCATION and os.path.exists(path):
        print('Loading model from file at %s' % path)
        net = torch.load(path).cuda()
    else:
        print('No saved model found, training a new one.')
        net = train_network()
    return net
