import matplotlib.pyplot as plt
import random
import math
import matplotlib.colors as mcolors
import numpy as np

MAX_ITERATIONS = 50

# https://cs.joensuu.fi/sipu/datasets/
#####################################################
class point(object):
    x = 0.0
    y = 0.0

    def __init__(self, x, y):
        self.x = x
        self.y = y

    @property
    def getX(self):
        return self.X

    @property
    def getY(self):
        return self.Y

class dataset(object):
    x_axis = []
    y_axis = []
    points = []
    numFeatures = 0

    def __init__(self, listX, listY):
        self.x_axis = listX
        self.y_axis = listY
        self.numFeatures = self.x_axis.__len__()

    def __init__(self, points):
        self.points = points
        self.numFeatures = self.points.__len__()

    @property
    def getNumfeatures(self):
        return self.features

# http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf
def kmeansplusplus(dataSet, k):
    pointList = [i for i in range(0, dataSet.numFeatures)]
    randomIndex = random.randint(0, dataSet.numFeatures)
    centroidList = []
    centroidList.append(point(dataSet.points[randomIndex].x, dataSet.points[randomIndex].y))
    distanceList = [999999999.0 for i in range(0,dataSet.numFeatures)]
    distanceList[randomIndex] = 0.0
    while (k > 1):
        pointList.remove(randomIndex)
        sumDis = 0.0
        for i in range(0, pointList.__len__()):
            distance = pow(abs(dataSet.points[randomIndex].x-dataSet.points[i].x),2)+pow(abs(dataSet.points[randomIndex].y-dataSet.points[i].y),2)
            distanceList[pointList[i]] = min(distanceList[pointList[i]], distance)
            sumDis += distanceList[pointList[i]]
        probabilityList = [distanceList[i]/sumDis for i in range(0, dataSet.numFeatures)]
        selectFloat = random.uniform(0, 1)
        sumProb = 0.0
        for i in range(0, dataSet.numFeatures):
            sumProb += probabilityList[i]
            if(sumProb>selectFloat):
                randomIndex = i
                distanceList[i] = 0.0
                centroidList.append(point(dataSet.points[randomIndex].x, dataSet.points[randomIndex].y))
                break
        k -= 1
    return centroidList

def shouldStop(oldCentroids, centroids, iterations):
    if iterations > MAX_ITERATIONS: return True
    return oldCentroids == centroids


def reassignToNearestCentroid(dataSet, centroids):
    label = []
    for i in range(0, dataSet.numFeatures):
        distance = 0.0
        oldDistance = 999999999.0
        thisLabel = 0
        for j in range(0, centroids.__len__()):
            abs_x = abs(dataSet.points[i].x - centroids[j].x)
            abs_y = abs(dataSet.points[i].y - centroids[j].y)
            distance = math.sqrt(pow(abs_x, 2) + pow(abs_y, 2))
            if distance < oldDistance:
                thisLabel = j
                oldDistance = distance
        label.append(thisLabel)

    return label


def getLabels(dataSet, centroids):
    label = reassignToNearestCentroid(dataSet, centroids)
    return label


def computeCentroids(dataSet, labels, k):
    centroids = []
    numLabels = []
    for i in range(0, k):
        centroids.append(point(0.0, 0.0))
        numLabels.append(0)
    for j in range(0, dataSet.numFeatures):
        thisLabel = labels[j]
        centroids[thisLabel].x += dataSet.points[j].x
        centroids[thisLabel].y += dataSet.points[j].y
        numLabels[thisLabel] += 1

    for k in range(0, k):
        centroids[k].x /= numLabels[k]
        centroids[k].y /= numLabels[k]

    return centroids


def getCentroids(dataSet, labels, k):
    newCentroids = computeCentroids(dataSet, labels, k)
    return newCentroids


def kmeans(dataSet, k):
    # Initialize centroids randomly
    # numFeatures = dataset.numFeatures
    returnVal = []
    labels = []
    # centroids = getRandomCentroids(dataset, k)
    centroids = kmeansplusplus(dataSet, k)

    # Initialize book keeping vars.
    iterations = 0
    oldCentroids = None

    # Run the main k-means algorithm
    while not shouldStop(oldCentroids, centroids, iterations):
        # Save old centroids for convergence test. Book keeping.
        oldCentroids = centroids
        iterations += 1

        # Assign labels to each datapoint based on centroids
        labels = getLabels(dataSet, centroids)

        # Assign centroids based on datapoint labels
        centroids = getCentroids(dataSet, labels, k)

    # We can get the labels too by calling getLabels(dataSet, centroids)
    returnVal.append(centroids)
    returnVal.append(labels)
    return returnVal


def equalToSolution(centPoints, realPoints):
    for i in range(0, centPoints.__len__()):
        if centPoints[i].x != realPoints[i].x or centPoints[i].y != realPoints[i]:
            return False
    return True

def computeDistance(x1, y1, x2, y2):
    return math.sqrt(pow(abs(x1-x2),2)+pow(abs(y1-y2),2))

def computeError(centPoints, realPoints, k):
    error = 0.0
    closestDistance = 999999999.0
    for i in range(0, k):
        distance = computeDistance(centPoints[i].x, centPoints.y, realPoints.x, realPoints.y)
        if distance < closestDistance:
            closestDistance = distance
        error += closestDistance

    return error

def computeError2(centPoints, realPoints, k):
    error = 0
    realPointsX = 0.0
    centPointsX = 0.0
    realPointsY = 0.0
    centPointsY = 0.0
    closestDistance = 999999999.0
    for i in range(0, k):
        realPointsX += centPoints[i].x
        realPointsY += centPoints[i].y
        centPointsX += realPoints[i].x
        centPointsY += realPoints[i].y
    error += abs(realPointsX-centPointsX) + abs(realPointsY - centPointsY)
    return error




############ MAIN ###########


# filename = ["cluster-n3000-k20.txt","a1-ga-cb.txt"]
# filename = ["cluster-n5250-k35.txt","a2-ga-cb.txt"]
filename = ["cluster-n7500-k50.txt","a3-ga-cb.txt"]
# options = [20, 30]
# options = [35, 50]
options = [50, 30]
maps = []
points = []
realPoints = []
k = options[0]
with open(filename[0]) as f:
    for line in f:  # Line is a string
        # split the string on whitespace, return a list of numbers
        # (as strings)
        numbers_str = line.split()
        # convert numbers to floats
        numbers_float = map(float, numbers_str)  # map(float,numbers_str) works too
        maps.append(numbers_float)
        points.append(point(numbers_float[0]/1000, numbers_float[1]/1000))
f.close()

with open(filename[1]) as f:
    for line in f:  # Line is a string
        # split the string on whitespace, return a list of numbers
        # (as strings)
        numbers_str = line.split()
        # convert numbers to floats
        numbers_float = map(float, numbers_str)  # map(float,numbers_str) works too
        maps.append(numbers_float)
        realPoints.append(point(numbers_float[0]/1000, numbers_float[1]/1000))
f.close()

ds = dataset(points)
cmap = plt.get_cmap('gnuplot')
colors = [cmap(i) for i in np.linspace(0, 1, k)]
colors2 = dict(**mcolors.CSS4_COLORS)
by_hsv = ((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                for name, color in colors2.items())
sorted_names = [name for hsv, name in by_hsv]


bestPoints = []
labels = []
iter = 0
error = 99999999.0
print "It is done "+str(options[1]) + " times for best results (it may be faster if you are lucky).\n"
while iter < options[1]:
    print "Perform clustering ... " + str(iter+1) + "th attempt."
    returnVal = kmeans(ds, k)
    print "Clustering is over. Now calculate the error."
    # finalCentroid = returnVal[0]
    # centX = []
    # centY = []
    centPoints = returnVal[0]
    # for m in range(0, finalCentroid.__len__()):
    #     centX.append(finalCentroid[m][0])
    #     centY.append(finalCentroid[m][1])
    if equalToSolution(centPoints, realPoints): break
    solutionError = computeError2(centPoints, realPoints, k)
    if solutionError < error:
        error = solutionError
        bestPoints = centPoints
        labels = returnVal[1]
    print "The error has been calculated. The difference with the solution is "+ str(error) + ".\n\n"
    iter+=1
print "Done! I will be show the result."

labelList = []
for i in range(0, k):
    labelList.append([])
for i in range(0,ds.numFeatures):
    labelList[labels[i]].append(i)
for i in range(0, k):
    for j in range(0, labelList[i].__len__()):
        index = labelList[i][j]
        plt.scatter(points[index].x, points[index].y, color = colors2[sorted_names[i]], marker='.')
for i in range(0, k):
    plt.scatter(bestPoints[i].x, bestPoints[i].y, c='black', marker='*')

plt.show()
