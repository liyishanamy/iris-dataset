#Yishan Li 10182827

import numpy  as np
import math
class peceptron:
    def __init__(self, leanrningRate, fireThreshold,epoch):
        self.trainingData = self.getTrainingData("iris_train.txt")
        self.trainingDataDesiredOutput = self.getDesiredOutput("iris_train.txt")
        self.testingData = self.getTrainingData("iris_test.txt")
        self.testingDataDesiredOutput = self.getDesiredOutput("iris_test.txt")
        self.weight = np.random.normal(-1,1,size=(5,3))
        self.optimumWeight = None
        self.learningRate = leanrningRate
        self.fireThreshold = fireThreshold
        self.epoch = epoch
        self.maxSuccessCount = 0
    # Obtain the training dataset.
    def getTrainingData(self, fileName):
        with open(fileName) as textFile:
            dataSet = [line.strip('\n').split(',') for line in textFile]
        for i in range(len(dataSet)):
            for j in range(len(dataSet[0])):
                if j == 4:
                    dataSet[i][j] = 1
                else:
                    dataSet[i][j] = float(dataSet[i][j])
        return dataSet

    # Get the desired output.
    def getDesiredOutput(self, fileName):
        desiredData=[]
        with open(fileName) as textFile:
            dataSet = [line.strip('\n').split(',') for line in textFile]
        for i in range(len(dataSet)):
            if dataSet[i][4] =="Iris-setosa":
                desiredData.append([1,0,0])
            if dataSet[i][4] =="Iris-versicolor":
                desiredData.append([0,1,0])
            if dataSet[i][4] =="Iris-virginica":
                desiredData.append([0,0,1])
        return desiredData

    # Activation function for training dataset. The activation function return 1 if the result is greater than threshold, otherwise return 0.
    def outputFuction(self, input, threshhold):
        result = []
        for value in input:
            output = 1/(1+(math.e)**(-value))
            if output >= threshhold:
                result.append(1)
            else:
                result.append(0)
        return result
    # Activation function for testing dataset. The activation function return 1 if the output is the maximum value within the result.
    def activationForTesting(self, input):
        result = []
        for value in input:
            output = 1 / (1 + (math.e) ** (-value))
            result.append(output)
        maxValue = max(result)
        for i in range(len(result)):
            if result[i] != maxValue:
                result[i] = 0
            else:
                result[i] = 1
        return result

    # get the summation of the training dataset and the weight vector.
    def getSummation(self, dataset, weight):
        summationVector = np.dot(dataset, weight)
        return summationVector

    # Apply pocket algorithm to my ANN model and try to get the optimum weight vector.
    def checkSuccess(self,inputData, desiredData):
        countSuccess = 0
        for i in range(len(inputData)):
            summation = self.getSummation(inputData[i], self.weight)
            actualResult = self.outputFuction(summation, self.fireThreshold)
            error = np.subtract(desiredData[i], actualResult)
            # if error vector is [0,0,0], then no error is found.
            if not np.any(error):
                countSuccess = countSuccess + 1
            #  if error vector is not [0,0,0], then error is found.
            else:
                # If the number of success is found greater than the recorded one, then the optimum vector also needs updating.
                if countSuccess > self.maxSuccessCount:
                    self.maxSuccessCount = countSuccess
                    self.optimumWeight = self.weight
                self.updateWeight(self.learningRate, error, inputData[i])
                # counter for success need setting back to 0 to keep track of the best length of run.
                countSuccess = 0

    # Update the weight vector based on the error, and the input data.
    def updateWeight(self, learningRate, error, inputDataset):
        error = np.transpose(error)
        deltaWeight = learningRate * np.dot(np.expand_dims(error,1), np.expand_dims(inputDataset,0))
        self.weight = self.weight + np.transpose(deltaWeight)

    # Train the model 1000 iterations and come up with an optimum weight vector.
    def train(self, dataset, desiredOutput, epoch):
        i = 0
        while i < epoch:
            # Check the number of success within each iteration
            self.checkSuccess(dataset,desiredOutput)
            i=i+1

    # Write the predictions on testing data in the "iris_result.txt" file
    def writeReport(self):
        # Get all the perdictions of the testing dataset.
        testData = self.testingData
        file = open("iris_result.txt", "w")
        for i in range(len(testData)):
            for j in range(len(testData[0])-1):
                file.write(str(testData[i][j]) + ",")
            # Change the prediction into a proper text format.
            self.checkCategory(file, i)
        file.close()

    # Check the category of the iris and write the corresponding class type into the file.
    def checkCategory(self,file,i):
        allActualResult = self.testData()
        if allActualResult[i] == [1, 0, 0]:
            file.write("Iris-setosa\n")
        if allActualResult[i] == [0, 1, 0]:
            file.write("Iris-versicolor\n")
        if allActualResult[i] == [0, 0, 1]:
            file.write("Iris-virginica\n")

    # Get the confusion matrix of the testing data or training data
    def defineConfusionMatrix(self, dataset,classLength):
        #for testing data result.
        if classLength == 10:
            actualResult = self.testData()
        if classLength == 40:
            actualResult = self.predictTrainingData()

        confusionMatrix = np.zeros((4, 4))
        for i in range(len(dataset)):
            if i < classLength:
                if actualResult[i] == [1,0,0]:
                    confusionMatrix[0][0] = confusionMatrix[0][0] + 1
                if actualResult[i] == [0,1,0]:
                    confusionMatrix[0][1] = confusionMatrix[0][1] + 1
                if actualResult[i] == [0,0,1]:
                    confusionMatrix[0][2] = confusionMatrix[0][2] + 1
            if((confusionMatrix[0][0]+confusionMatrix[0][1]+confusionMatrix[0][2]) == 0):
                confusionMatrix[0][3] = 0
            else:
                confusionMatrix[0][3] = confusionMatrix[0][0]/(confusionMatrix[0][0]+confusionMatrix[0][1]+confusionMatrix[0][2])

            if (i>=classLength and i<2*classLength):
                if actualResult[i] == [1,0,0]:
                   confusionMatrix[1][0] = confusionMatrix[1][0] + 1
                if actualResult[i] == [0,1,0]:
                    confusionMatrix[1][1] = confusionMatrix[1][1] + 1
                if actualResult[i] == [0,0,1]:
                    confusionMatrix[1][2] = confusionMatrix[1][2] + 1
            if(confusionMatrix[1][0]+confusionMatrix[1][1]+confusionMatrix[1][2] == 0):
                confusionMatrix[1][3] = 0
            else:
                confusionMatrix[1][3] = confusionMatrix[1][1]/(confusionMatrix[1][0]+confusionMatrix[1][1]+confusionMatrix[1][2])
            if (i>=2*classLength):
                if actualResult[i] == [1,0,0]:
                   confusionMatrix[2][0] = confusionMatrix[2][0] + 1
                if actualResult[i] == [0,1,0]:
                    confusionMatrix[2][1] = confusionMatrix[2][1] + 1
                if actualResult[i] == [0,0,1]:
                    confusionMatrix[2][2] = confusionMatrix[2][2] + 1
            if(confusionMatrix[2][0]+confusionMatrix[2][1]+confusionMatrix[2][2]==0):
                confusionMatrix[2][3] = 0
            else:
                confusionMatrix[2][3] = confusionMatrix[2][2]/(confusionMatrix[2][0]+confusionMatrix[2][1]+confusionMatrix[2][2])
        for j in range(4):
            if j == 0:
                confusionMatrix[3][0] = confusionMatrix[0][0]/(confusionMatrix[0][0] + confusionMatrix[1][0] +confusionMatrix[2][0])
            if j == 1:
                confusionMatrix[3][1] = confusionMatrix[1][1]/(confusionMatrix[0][1] + confusionMatrix[1][1] +confusionMatrix[2][1])
            if j == 2:
                confusionMatrix[3][2] = confusionMatrix[2][2]/(confusionMatrix[0][2] + confusionMatrix[1][2] +confusionMatrix[2][2])
            if j ==3:
                confusionMatrix[3][3] = (confusionMatrix[0][0]+confusionMatrix[1][1]+confusionMatrix[2][2])/ (confusionMatrix[0][0]+confusionMatrix[0][1]+confusionMatrix[0][2] + confusionMatrix[1][0]+confusionMatrix[1][1]+confusionMatrix[1][2] +confusionMatrix[2][0]+confusionMatrix[2][1]+confusionMatrix[2][2])
        return confusionMatrix

    # Get the training data prediction using the optimum weight.
    def predictTrainingData(self):
        traingData = self.trainingData
        allActualResult = []
        for i in range(len(traingData)):
            summation = self.getSummation(traingData[i], self.optimumWeight)
            actualResult = self.activationForTesting(summation)
            allActualResult.append(actualResult)
        return allActualResult


    # Use the trained model to predict the outcome for the testing dataset.
    def testData(self):
        successCount = 0
        testData = self.testingData
        desireData = self.testingDataDesiredOutput
        allActualResult = []
        for i in range(len(testData)):
            summation = self.getSummation(testData[i], self.optimumWeight)
            actualResult = self.activationForTesting(summation)
            allActualResult.append(actualResult)
            error = np.subtract(desireData[i], actualResult)
            if not np.any(error):
                successCount = successCount + 1
        return allActualResult

def main():
    # set the learning rate to be 0.0001.
    # set the activation fire threshold to be 0.5
    # set the epoch to be 10000
    perceptron=peceptron(0.0001,0.5,10000)
    # train the model based on the training data.
    perceptron.train(perceptron.trainingData,perceptron.trainingDataDesiredOutput, perceptron.epoch)
    print("optimal weight:\n", perceptron.optimumWeight)
    perceptron.writeReport()
    print("testing dataset confusion matrix\n", perceptron.defineConfusionMatrix(perceptron.testingData,10))
    print("training dataset confusion matrix\n", perceptron.defineConfusionMatrix(perceptron.trainingData, 40))

main()