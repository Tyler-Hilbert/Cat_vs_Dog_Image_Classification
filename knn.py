# KNN for fruit classification
import math

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
    nearestDistance = 99999
    nearestNeighbor = "UNDEF"
    for key in data:
        for point in data[key]:
            distance = getDistance(testPoint, point)
            ## print ("Distance", distance)
            if distance < nearestDistance:
                nearestDistance = distance
                nearestNeighbor = key
    return nearestNeighbor



#####################################
incorrect = 0
correct = 0

for testRowNum in range(1, 60): # Loop through each data point and test KNN
    data = {}

    # Test data piece
    line = open('fruit_data_with_colors.txt', 'r').readlines()[testRowNum]
    row = line.split("\t")
    testPoint = (float(row[3]), float(row[4]), float(row[5]), float(row[6]))
    testClass = row[1]
    ## print (testPoint)

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


    nearestNeighbor = knn(testPoint, data)
    ## print ("Found nearest neighbor", nearestNeighbor)
    print (nearestNeighbor, testClass)
    if (nearestNeighbor == testClass):
        print ("Correct")
        correct += 1
    else:
        print ("Incorrect")
        incorrect += 1


print ("Correct", correct)
print ("Incorrect", incorrect)
print ("Accuracy", (correct/(correct+incorrect)))
