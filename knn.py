# KNN for fruit classification

## Results:
## K=1: Accuracy 66%
## K=3: Accuracy 73%
## K=5: Accuracy 61%
## K=7: Accuracy 51%

import math



##################################### Constants
## Index values for data points to test as target point. Note this is NOT the training set, all points are in the training set except for the one under test
START_INDEX = 1
END_INDEX = 59
K = 3

##################################### Functions
# Calculates distance between tuples
def getDistance(point1, point2):
    if len(point1) != len(point2):
        error ("Points don't have same size")

    diffSquared = 0
    for p1, p2 in zip(point1, point2):
        diffSquared += ( (p1 - p2) ** 2 )

    rms = math.sqrt(diffSquared)
    return rms

# Get nearest neighbor using KNN
def knn(testPoint, data):
    nearestDistance = [99999]*K
    nearestNeighbor = ["UNDEF"]*K
    for key in data: # Loop through each label
        for point in data[key]: # Loop through each data point
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
    return nearestNeighbor



##################################### Main
incorrect = 0
correct = 0

for testRowNum in range(START_INDEX, END_INDEX+1): # Loop through each data point and test KNN
    data = {}

    # Test data piece
    line = open('fruit_data_with_colors.txt', 'r').readlines()[testRowNum]
    row = line.split("\t")
    testPoint = (float(row[3]), float(row[4]), float(row[5]), float(row[6]))
    testClass = row[1]
    print ("Test Point:", testPoint)

    # Read in "training" data
    for line in open('fruit_data_with_colors.txt', 'r').readlines()[1:]:
        line = line.strip()
        row = line.split("\t")

        name = row[1]
        point = (float(row[3]), float(row[4]), float(row[5]), float(row[6]))
        if point == testPoint:
            continue

        if name in data:
            data[name].append ( point )
        else:
            data[name] = [ point ]


    nearestNeighbors = knn(testPoint, data)
    print ("nearestNeighbors", nearestNeighbors)
    prediction = max(set(nearestNeighbors), key=nearestNeighbors.count)
    print ("prediction", prediction)

    if (prediction == testClass):
        print ("Correct")
        correct += 1
    else:
        print ("Incorrect")
        incorrect += 1


print ("Correct", correct)
print ("Incorrect", incorrect)
print ("Accuracy", (correct/(correct+incorrect)))
