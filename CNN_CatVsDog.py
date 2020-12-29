# CNN for classifying image as dog or cat
# Data - https://www.kaggle.com/c/dogs-vs-cats/data
# Results - 91% (AlexNet)

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
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.001



########## Neural Network
class ConvNN(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super(ConvNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

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
