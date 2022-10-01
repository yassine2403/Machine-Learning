import csv

with open(r'C:\Users\Fehmi Laourine\Downloads\iris.data.txt') as csvfile:

  lines = csv.reader(csvfile)
  for row in lines :

      print (', '.join(row))


import random

def loadDataset(filename, split, trainingSet=[] , testSet=[]):

 with open(filename, 'r') as csvfile:

  lines = csv.reader(csvfile)

  dataset = list(lines)

  for x in range(len(dataset)-1):

    for y in range(4):

      dataset[x][y] = float(dataset[x][y])

    if random.random() < split:

      trainingSet.append(dataset[x])

    else:

       testSet.append(dataset[x])
    


import math
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

import operator

def getNeighbors(trainingSet, testInstance, k):

  distances = []

  length = len(testInstance)-1

  for x in range(len(trainingSet)):

     dist = euclideanDistance(testInstance, trainingSet[x], length)

     distances.append((trainingSet[x], dist))

     distances.sort(key=operator.itemgetter(1))

     neighbors = []

  for x in range(k):

     neighbors.append(distances[x][0])

     return neighbors


import operator

def getResponse(neighbors):

  classVotes = {}

  for x in range(len(neighbors)):

     response = neighbors[x][ -1 ] #complete with appropriate number

     if response in classVotes:

       classVotes[response] += 1
     else:
       classVotes[response] = 1

       sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)

  return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
      if testSet[x][-1] in predictions[x]: 
          correct = correct + 1

    return (correct/float(len(testSet))*100) 


def main():
    # prepare data
    trainingSet=[]
    testSet=[]
    split = 0.70
    file=r'C:\Users\Fehmi Laourine\Downloads\iris.data.txt'
    loadDataset(file, split, trainingSet, testSet)
    print ('Train set: ' + repr(len(trainingSet)))
    print ('Test set: ' + repr(len(testSet)))
    # generate predictions
    predictions=[]
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')
main()
