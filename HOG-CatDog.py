# Histogram of Oriented Gradients + SVM for classifying image as dog or cat
# Data - https://www.kaggle.com/c/dogs-vs-cats/data
# HOG reference - https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html
# SVM reference - https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python

'''
Results:
NUM_TEST_SAMPLES_PER = 25, NUM_TRAIN_SAMPLES_PER = 100
accuracy = 0.56

NUM_TEST_SAMPLES_PER = 25, NUM_TRAIN_SAMPLES_PER = 1000
accuracy = 0.68

NUM_TEST_SAMPLES_PER = 25, NUM_TRAIN_SAMPLES_PER = 5000
accuracy = 0.58

NUM_TEST_SAMPLES_PER = 25, NUM_TRAIN_SAMPLES_PER = 10000
accuracy = 0.56
'''

from skimage.feature import hog
from skimage import data, exposure
from PIL import Image
import numpy as np
from sklearn import svm
from os import listdir
from os.path import isfile, join
import random

##################################### Constants
NUM_TEST_SAMPLES_PER = 25 # Number of test images to set aside for each class
NUM_TRAIN_SAMPLES_PER = 5000 # Number of training images to read
DATA_PATH = "data/"
DOG_LABEL = 'dog' # 0
CAT_LABEL = 'cat' # 1


##################################### Functions
# Performs HOG
def getHistogram(image):
    fd, hogImage = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True, multichannel=True)
    fd = fd.round(1)
    hist = []
    hist.append(np.count_nonzero(fd == 0)/fd.size)
    hist.append(np.count_nonzero(fd == 0.1)/fd.size)
    hist.append(np.count_nonzero(fd == 0.2)/fd.size)
    hist.append(np.count_nonzero(fd == 0.3)/fd.size)
    hist.append(np.count_nonzero(fd == 0.4)/fd.size)
    hist.append(np.count_nonzero(fd == 0.5)/fd.size)
    hist.append(np.count_nonzero(fd == 0.6)/fd.size)
    hist.append(np.count_nonzero(fd == 0.7)/fd.size)
    hist.append(np.count_nonzero(fd == 0.8)/fd.size)
    hist.append(np.count_nonzero(fd == 0.9)/fd.size)
    hist.append(np.count_nonzero(fd == 1)/fd.size)
    return np.array(hist)


# Gets image
def getImg(imgPath):
    img = Image.open(imgPath)
    return img


##################################### Main

# Read Data
# TODO - This is an overly complicated way to read the data which needs to be improved.
#   Also it only reads the first N number of samples rather than a random N number of samples.
#   The N samples are randomized into train and test.
catCount = 0
dogCount = 0
data = {DOG_LABEL: [], CAT_LABEL: []}
imgFiles = [f for f in listdir(DATA_PATH) if isfile(join(DATA_PATH, f))]
for imgFile in imgFiles:
    print ("dogCount", dogCount, "catCount", catCount)
    imgFile = DATA_PATH + imgFile
    if CAT_LABEL in imgFile:
        if catCount < (NUM_TEST_SAMPLES_PER + NUM_TRAIN_SAMPLES_PER):
            data[CAT_LABEL].append(getHistogram(getImg(imgFile)))
            catCount += 1
    elif DOG_LABEL in imgFile:
        if dogCount < (NUM_TEST_SAMPLES_PER + NUM_TRAIN_SAMPLES_PER):
            data[DOG_LABEL].append(getHistogram(getImg(imgFile)))
            dogCount += 1
    else:
        error ("Not a valid label")

# Split data
testData = {CAT_LABEL: [], DOG_LABEL: []}
random.shuffle(data[CAT_LABEL])
random.shuffle(data[DOG_LABEL])
for i in range(NUM_TEST_SAMPLES_PER):
    testData[CAT_LABEL].append(data[CAT_LABEL].pop(0))
    testData[DOG_LABEL].append(data[DOG_LABEL].pop(0))

# Convert dictionary into list data structure
XTrain = []
yTrain = []
XTest = []
yTest = []
for X in data[DOG_LABEL]:
    XTrain.append(X)
    yTrain.append(0)
for X in data[CAT_LABEL]:
    XTrain.append(X)
    yTrain.append(1)
for X in testData[DOG_LABEL]:
    XTest.append(X)
    yTest.append(0)
for X in testData[CAT_LABEL]:
    XTest.append(X)
    yTest.append(1)

# Train SVM
clf = svm.SVC(kernel='linear')
clf.fit(XTrain, yTrain)

# Test / Print Results
yPred = clf.predict(XTest)
correct = 0
for p, t in zip(yPred, yTest):
    if p == t:
        correct += 1
    print ("pred", p, "test", t)
accuracy = correct / (2*NUM_TEST_SAMPLES_PER)
print ("accuracy", accuracy)
