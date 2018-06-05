"""use resnet152 as the model instead of the tutorial one"""
import ast
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.models import resnet152

CLASSES_LOCATION = 'data/imagenet.classes.txt'

with open(CLASSES_LOCATION) as file:
    CLASSES = ast.literal_eval(file.read())

SIZE_TRANSFORM = transforms.Compose([
    transforms.Resize(240),
    transforms.CenterCrop(224)
    ])

PREPROCESSING_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])


def get_network():
    """returns the pretrained resnet152 network"""
    return resnet152(pretrained=True).eval().cuda()


def predict_category(image, net):
    """return the predicted category of the image"""
    image = Variable(image.unsqueeze(0).cuda())
    outputs = net(image)
    _, predicted = torch.max(outputs.data, 1) # pylint: disable=no-member
    return predicted[0]
