# CNN for classifying image as dog or cat
# Data - https://www.kaggle.com/c/dogs-vs-cats/data
# Results - 89.5%

import os
import numpy as np
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib

########## Constants
TRAIN_PATH = "dataYouTubeFormat/train"
TEST_PATH = "dataYouTubeFormat/test"
BATCH_SIZE = 256
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001



########## Neural Network
class ConvNN(nn.Module):
    def __init__(self, num_classes=6):
        super(ConvNN, self).__init__()


        # ((w - f + 2p) / s) + 1
        #   w = height or width
        #   f = kernel size
        #   p = padding
        #   s = stride

        # Shape: (batch size, number of channels, image height, image width)
        # Shape: (256, 3, 150, 150)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=20)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=28, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=28)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=28, out_channels=36, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=36)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(in_channels=36, out_channels=42, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=42)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(in_channels=42, out_channels=50, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(num_features=50)
        self.relu6 = nn.ReLU()
        self.conv7 = nn.Conv2d(in_channels=50, out_channels=58, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(num_features=58)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(in_channels=58, out_channels=66, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(num_features=66)
        self.relu8 = nn.ReLU()
        self.conv9 = nn.Conv2d(in_channels=66, out_channels=74, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(num_features=74)
        self.relu9 = nn.ReLU()
        self.conv10 = nn.Conv2d(in_channels=74, out_channels=82, kernel_size=3, stride=1, padding=1)
        self.bn10 = nn.BatchNorm2d(num_features=82)
        self.relu10 = nn.ReLU()
        self.conv11 = nn.Conv2d(in_channels=82, out_channels=90, kernel_size=3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm2d(num_features=90)
        self.relu11 = nn.ReLU()

        self.fc = nn.Linear(in_features=90*75*75, out_features=num_classes)

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(output)
        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)
        output = self.conv4(output)
        output = self.bn4(output)
        output = self.relu4(output)
        output = self.conv5(output)
        output = self.bn5(output)
        output = self.relu5(output)
        output = self.conv6(output)
        output = self.bn6(output)
        output = self.relu6(output)
        output = self.conv7(output)
        output = self.bn7(output)
        output = self.relu7(output)
        output = self.conv8(output)
        output = self.bn8(output)
        output = self.relu8(output)
        output = self.conv9(output)
        output = self.bn9(output)
        output = self.relu9(output)
        output = self.conv10(output)
        output = self.bn10(output)
        output = self.relu10(output)
        output = self.conv11(output)
        output = self.bn11(output)
        output = self.relu11(output)

        output = output.view(-1, 90*75*75)

        output = self.fc(output)

        return output

########## Script
device = torch.device ('cuda' if torch.cuda.is_available() else 'cpu')
print (device)

transformer = transforms.Compose([
    transforms.Resize( (150,150) ), # Is this the correct size?
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainLoader = DataLoader (
    torchvision.datasets.ImageFolder(TRAIN_PATH, transform=transformer),
    batch_size=BATCH_SIZE,
    shuffle=True
)

testLoader = DataLoader (
    torchvision.datasets.ImageFolder(TEST_PATH, transform=transformer),
    batch_size=BATCH_SIZE,
    shuffle=True
)

root = pathlib.Path(TRAIN_PATH)
classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])
classes.remove(".DS_Store") # For OSX
print (classes)

trainCount = len(glob.glob(TRAIN_PATH + '/**/*.jpg'))
testCount = len(glob.glob(TEST_PATH + '/**/*.jpg'))

print ("trainCount", trainCount, "testCount", testCount)

model = ConvNN(num_classes=6).to(device)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
lossFunction = nn.CrossEntropyLoss()

print (model.__dict__)

bestAccuracy = 0
for epoch in range(NUM_EPOCHS):
    model.train()
    trainAccuracy = 0
    trainLoss = 0

    for i, (images, labels) in enumerate (trainLoader):
        if torch.cuda.is_available():
            images = Variable (images.cuda())
            labels = Variable (labels.cuda())

        optimizer.zero_grad()

        outputs = model(images)
        loss = lossFunction(outputs, labels)
        loss.backward()
        optimizer.step()

        trainLoss += loss.cpu().data*images.size(0)
        _,prediction = torch.max(outputs.data, 1)

        trainAccuracy += int(torch.sum(prediction==labels.data))

    trainAccuracy = trainAccuracy / trainCount
    trainLoss = trainLoss / trainCount

    model.eval()

    testAccuracy = 0
    for i, (images, labels) in enumerate (testLoader):
        if torch.cuda.is_available():
            images = Variable (images.cuda())
            labels = Variable (labels.cuda())

        outputs = model(images)
        _,prediction = torch.max(outputs.data, 1)
        testAccuracy += int(torch.sum(prediction==labels.data))

    testAccuracy = testAccuracy / testCount

    print ("epoch", epoch, "testAccuracy", testAccuracy, "trainAccuracy", trainAccuracy)


    if testAccuracy > bestAccuracy:
        torch.save(model.state_dict(), 'best_checkpoint.model')
        bestAccuracy = testAccuracy
