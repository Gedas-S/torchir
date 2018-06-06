import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet152

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(240),
    transforms.RandomResizedCrop(224),
])

SIZE_TRANSFORM = transforms.Compose([
    transforms.Resize(240),
    transforms.CenterCrop(224)
])

PREPROCESSING_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

MODEL_LOCATION = 'model/transfer.model'
DATA_LOCATION = 'data/hymenoptera_data/train'

CLASSES = (
    'ant',
    'bee',
)


def train_network():
    start_time = time.clock()
    # define network
    net = resnet152(pretrained=True)
    for parameter in net.parameters():
        parameter.required_grad = False
    net.fc = nn.Linear(net.fc.in_features, 2)
    net.cuda()

    trainset = datasets.ImageFolder(
        DATA_LOCATION, 
        transforms.Compose([TRAIN_TRANSFORM, PREPROCESSING_TRANSFORM])
    )
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    lr = 0.001
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)  # stochastic gradient descent

    # train
    for _ in range(4):
        for _ in range(2):
            for data in trainloader:
                inputs, labels = data
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        lr *= 0.2
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # save model
    if MODEL_LOCATION:
        path = os.path.abspath(MODEL_LOCATION)
        print('Saving model to %s' % path)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(net, path)

    print('model trained in %sms' % (time.clock() - start_time))
    return net


def get_network():
    path = os.path.abspath(MODEL_LOCATION)
    if MODEL_LOCATION and os.path.exists(path):
        print('Loading model from file at %s' % path)
        net = torch.load(path).cuda()
    else:
        print('No saved model found, training a new one.')
        net = train_network()
    return net
