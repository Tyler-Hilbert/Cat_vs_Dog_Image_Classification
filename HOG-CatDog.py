# Histogram of Oriented Gradients + SVM for classifying image as dog or cat
# Data - https://www.kaggle.com/c/dogs-vs-cats/data
# HOG reference - https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html
# SVM reference - https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python

# Results in HOG-CatDog.results.txt
# Various parameters were played with, but on average it got around a 60% accuracy, with a slightly higher accuracy for dog than cat.

from skimage.feature import hog
from skimage import data, exposure
from PIL import Image
import numpy as np
from sklearn import svm
from os import listdir
from os.path import isfile, join
import random

##################################### Constants
NUM_TEST_SAMPLES_PER = 50 # Number of test images to set aside for each class
NUM_TRAIN_SAMPLES_PER = 5000 # Number of training images to read
DATA_PATH = "data/"
DOG_LABEL = 'dog' # 0
CAT_LABEL = 'cat' # 1


##################################### Functions
# Performs HOG
#  Return None if HOG can't be created
def getHistogram(image):
    fd, hogImage = hog(image, orientations=64, pixels_per_cell=(32, 32),
            cells_per_block=(1, 1), visualize=True, multichannel=True)

    if fd.size == 0:
        return None

    fd = fd.round(1)
    hist = []

    # Put into bins incremented from 0 to 1 inclusively
    for i in np.arange(0, 1.1, 0.1).round(1).tolist():
        hist.append(np.count_nonzero(fd == i)/fd.size)
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
    print ("imgFile", imgFile)
    imgFile = DATA_PATH + imgFile
    if CAT_LABEL in imgFile:
        if catCount < (NUM_TEST_SAMPLES_PER + NUM_TRAIN_SAMPLES_PER):
            parsedHistogram = getHistogram(getImg(imgFile))
            if parsedHistogram is None:
                continue
            else:
                data[CAT_LABEL].append(parsedHistogram)
                catCount += 1
    elif DOG_LABEL in imgFile:
        if dogCount < (NUM_TEST_SAMPLES_PER + NUM_TRAIN_SAMPLES_PER):
            parsedHistogram = getHistogram(getImg(imgFile))
            if parsedHistogram is None:
                continue
            else:
                data[DOG_LABEL].append(parsedHistogram)
                dogCount += 1
    else:
        print ("warning: Could not load file:", imgFile)

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
catCorrect = 0
dogCorrect = 0
for p, t in zip(yPred, yTest):
    if p == t:
        if p == 0:
            dogCorrect += 1
        else:
            catCorrect += 1
    print ("pred", p, "test", t)

dogAccuracy = dogCorrect / NUM_TEST_SAMPLES_PER
catAccuracy = catCorrect / NUM_TEST_SAMPLES_PER
print ("dogAccuracy", dogAccuracy)
print ("catAccuracy", catAccuracy)
