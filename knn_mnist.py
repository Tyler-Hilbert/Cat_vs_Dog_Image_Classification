# KNN for classifying MNIST handwritten digits

from os import listdir
from os.path import isfile, join
import random
import math
from PIL import Image

'''
Results
K=3, 10 test images for each digit
0 accuracy 1.0
1 accuracy 1.0
2 accuracy 1.0
3 accuracy 0.9
4 accuracy 1.0
5 accuracy 0.9
6 accuracy 1.0
7 accuracy 1.0
8 accuracy 0.9
9 accuracy 0.9
Total accuracy 0.96
'''

##################################### Constants
K = 7 # How many nearest neighbors to use for classification
IMG_SIZE = 28 # Number of X and Y pixels in each image
DATA_DIR = "mnist/" # Directory of dataset
NUM_TEST_SAMPLES_PER = 10 # Number of test images to set aside for each digit

##################################### Main
def main():
    # Read Data
    data = {}
    for digit in range(0,10):
        digit = str(digit)
        data[str(digit)] = []
        PATH = DATA_DIR + digit + '/'
        imgFiles = [f for f in listdir(PATH) if isfile(join(PATH, f))]
        for imgFile in imgFiles:
            imgFile = PATH + imgFile
            data[digit].append(imgFile)

    ## print (data)

    # Split data
    testData = {}
    for digit in range(0,10):
        digit = str(digit)
        testData[digit] = []
        random.shuffle(data[digit])
        for i in range(NUM_TEST_SAMPLES_PER):
            testData[digit].append(data[digit].pop(0))

    # Test
    correctCount = {}
    for digit in range(0,10):
        digit = str(digit)
        correctCount[digit] = 0

    for testKey in testData:
        for testImg in testData[testKey]:
            print ("testImg:", testImg)
            nearestNeighbors = knn(testImg, data)
            print ("nearestNeighbors", nearestNeighbors)
            prediction = max(set(nearestNeighbors), key=nearestNeighbors.count)
            if prediction == testKey:
                correctCount[str(testKey)] += 1
            print ("*"*20)


    # Print results
    totalCorrect = 0
    for digit in range(0,10):
        digit = str(digit)
        print (digit, "accuracy", correctCount[digit]/NUM_TEST_SAMPLES_PER)
        totalCorrect += correctCount[digit]
    print ("Total accuracy", totalCorrect/(NUM_TEST_SAMPLES_PER*10))

##################################### Functions
# Get nearest neighbor using KNN
def knn(testPoint, data):
    nearestDistance = [99999]*K
    nearestNeighbor = ["UNDEF"]*K
    dataNum = 0
    for key in data: # Loop through each label
        for point in data[key]: # Loop through each data point
            dataNum += 1
            # Calcaulte distance and check if it is nearest
            distance = getDistance(testPoint, point)
            ## print ("Distance", distance)
            if distance < nearestDistance[0]: # Replace
                nearestDistance[0] = distance
                nearestNeighbor[0] = key
                # Sort
                zippedData = zip(nearestDistance, nearestNeighbor)
                sortedPairs = sorted(zippedData, reverse=True)
                tuples = zip(*sortedPairs)
                nearestDistance, nearestNeighbor = [ list(tuple) for tuple in  tuples]
                ###print ("nearestNeighbor", nearestNeighbor, "nearestDistance", nearestDistance, "dataNum", dataNum)
    return nearestNeighbor

# Calculates distance between tuples
def getDistance(img1Path, img2Path):
    img1 = getImg(img1Path)
    img2 = getImg(img2Path)
    diffSquared = 0
    for x in range (0, IMG_SIZE):
        for y in range (0, IMG_SIZE):
            cord = x, y
            diffSquared += ( (img1.getpixel(cord) - img2.getpixel(cord)) ** 2 )
    rms = math.sqrt(diffSquared)
    return rms

# Gets image and converts to a scaled black and white image
def getImg(imgPath):
    img = Image.open(imgPath)
    return img

if __name__ == "__main__":
    main()
