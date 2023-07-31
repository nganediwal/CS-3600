from NeuralNetUtil import buildExamplesFromCarData,buildExamplesFromPenData
from NeuralNet import buildNeuralNet
from math import pow, sqrt

def average(argList):
    return sum(argList)/float(len(argList))

def stDeviation(argList):
    mean = average(argList)
    diffSq = [pow((val-mean),2) for val in argList]
    return sqrt(sum(diffSq)/len(argList))

penData = buildExamplesFromPenData()
def testPenData(hiddenLayers = [24]):
    return buildNeuralNet(penData, maxItr = 200, hiddenLayerList = hiddenLayers)

carData = buildExamplesFromCarData()
def testCarData(hiddenLayers = [16]):
    return buildNeuralNet(carData, maxItr = 200,hiddenLayerList = hiddenLayers)

def printt(results):
    print("MAX:     {}".format(max(results)))
    print("AVERAGE: {}".format(average(results)))
    print("STD:     {}".format(stDeviation(results)))

def q5():
    pen_results = []
    for _ in range(5):
        pen_results.append(testPenData()[1])
    printt(pen_results)

    car_results = []
    for _ in range(5):
        car_results.append(testPenData()[1])
    printt(car_results)


def q6():
    for i in range(0, 41, 5):
        results = []
        for _ in range(5):
            results.append(testPenData([i])[1])
        print("PENDATA PERCEPTRON COUNT {}".format(i))
        printt(results) #get max, average, and standard deviation

    for i in range(0, 41, 5):
        results = []
        for _ in range(5):
            results.append(testCarData([i])[1])
        print("PENDATA PERCEPTRON COUNT {}".format(i))
        printt(results) #get max, average, and standard deviation


q5()
q6()