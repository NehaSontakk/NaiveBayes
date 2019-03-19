# STEP 1 : Handle Data
# The dummy dataset doesn't have any string values so we don't have to handle that.
# But we should split the dataset into test and train even though we don't have many values.
import random
def splitdataset(dataset):
	splitratio = 0.67						# 67% of the dataset will be used to train
	trainsize = int(len(dataset)*splitratio)
	trainset = []							# Keep the train values in this list
	copy = list(dataset)						# Create a copy of the dataset
	while len(trainset) < trainsize:				# Fill trainset with data till it reaches trainsize
		index = random.randrange(len(copy))			# decide random elements indexes to put in the trainset
		trainset.append(copy.pop(index))			# Remove decided index element from copy and put it in trainset
	return [trainset,copy]						# The copy is just left with the test elements
		

# STEP 2 : Summarize Data
# Seperate data by class and calculate Mean, Standard deviation.
# Then summarize dataset and attributes by class

def separatebyclass(dataset):						# seperate the dataset by class i.e if it belongs to 0 or 1
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]					# vector contains ith row
		if(vector[-1] not in separated):			# if the last column is not in the separated list
			separated[vector[-1]]=[]			# maintain as a seperate list and map values to the category values
		separated[vector[-1]].append(vector)
	return separated


import math
def calculatemean(values):						# calculate mean of all values
	return sum(values)/float(len(values))
	
def std_deviation(values):						# calculate the standard deviation
	mean = calculatemean(values)
	step1 = sum([(x-mean)**2 for x in values])
	step2 = step1/float(len(values)-1)
	std_dev = math.sqrt(step2)
	return std_dev
	
def summarize(dataset):
	summary = [(calculatemean(i),std_deviation(i)) for i in zip(*dataset)]	#zip makes own lists from the dataset 
	del summary[-1]							# remove last column of dataset (our 'y' values)
	return summary


def summarizebyclass(dataset):
	separated = separatebyclass(dataset)
	summaries = {}
	for classValue, instances in separated.iteritems():
		summaries[classValue] = summarize(instances)
	return summaries
	
# Here first calculate the gaussian probability density function
# When making predictions these parameters can be plugged into the Gaussian PDF with a new input for the variable
# In return the Gaussian PDF will provide an estimate of the probability of that new input value for that class.
# Gpdf(x, mean, sd) = (1 / (sqrt(2 * PI) * sd)) * exp(-((x-mean^2)/(2*sd^2)))
# for example for fruit apple with features shape,size,colour
# apple(yess) =  P(Gpdf(shape)|class=apple(yess)) * P(pdf(size)|class=apple(yess))*P(pdf(colour)|class=apple(yess))* P(class=apple)

def calculateGprobability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
	

# STEP 3: Making predic-shunnnnnnsss
# Calculate class probabilities
# The class probabilities are simply the frequency of instances that belong to each class divided by the total number of instances.
# P(class=1) = count(class=1) / (count(class=0) + count(class=1))

def classprob(summaries,inputvector):
	probabilities = {}
	for values,summaries in summaries.iteritems():			# For all values and summaries each class 0 and 1 in separated list
		probabilities[values]=1
		for i in range(len(summaries)):
			mean,std_dev = summaries[i]
			x = inputvector[i]
			probabilities[values] *= calculateGprobability(x, mean, std_dev)
	return probabilities 


# make a prediction

def predict(summaries, inputvector):
	probabilities = classprob(summaries, inputvector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel


# make predictions for test set data

def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions

# get accuracy
def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

# I'll create a dummy dataset 

dataset = [[6,148,72,35,0,33.6,0.627,50,1],
[1,85,66,29,0,26.6,0.351,31,0],
[8,183,64,0,0,23.3,0.672,32,1],
[1,89,66,23,94,28.1,0.167,21,0],
[0,137,40,35,168,43.1,2.288,33,1],
[5,116,74,0,0,25.6,0.201,30,0],
[3,78,50,32,88,31.0,0.248,26,1],
[10,115,0,0,0,35.3,0.134,29,0],
[2,197,70,45,543,30.5,0.158,53,1],
[8,125,96,0,0,0.0,0.232,54,1],
[4,110,92,0,0,37.6,0.191,30,0],
[10,168,74,0,0,38.0,0.537,34,1],
[10,139,80,0,0,27.1,1.441,57,0],
[1,189,60,23,846,30.1,0.398,59,1],
[5,166,72,19,175,25.8,0.587,51,1],
[7,100,0,0,0,30.0,0.484,32,1],
[0,118,84,47,230,45.8,0.551,31,1],
[7,107,74,0,0,29.6,0.254,31,1],
[1,103,30,38,83,43.3,0.183,33,0],
[1,115,70,30,96,34.6,0.529,32,1],
[3,126,88,41,235,39.3,0.704,27,0],
[8,99,84,0,0,35.4,0.388,50,0],
[7,196,90,0,0,39.8,0.451,41,1]]

train,test = splitdataset(dataset)
print("Number of elements in train and test set respectively : ",len(train),"&",len(test))

separated = separatebyclass(train)
print(separated)

summaries = summarize(dataset)
print(summaries)

summarizebycla = summarizebyclass(train)
print(summarizebyclass)


predictions = getPredictions(summarizebycla, test)
print(predictions)

accuracy = getAccuracy(test, predictions)
print(accuracy)


# OUTPUT
"""
'Number of elements in train and test set respectively : ', 15, '&', 8)
{0: [[10, 115, 0, 0, 0, 35.3, 0.134, 29, 0], [1, 89, 66, 23, 94, 28.1, 0.167, 21, 0], [1, 85, 66, 29, 0, 26.6, 0.351, 31, 0], [10, 139, 80, 0, 0, 27.1, 1.441, 57, 0], [5, 116, 74, 0, 0, 25.6, 0.201, 30, 0]], 1: [[0, 137, 40, 35, 168, 43.1, 2.288, 33, 1], [0, 118, 84, 47, 230, 45.8, 0.551, 31, 1], [8, 183, 64, 0, 0, 23.3, 0.672, 32, 1], [5, 166, 72, 19, 175, 25.8, 0.587, 51, 1], [2, 197, 70, 45, 543, 30.5, 0.158, 53, 1], [6, 148, 72, 35, 0, 33.6, 0.627, 50, 1], [7, 196, 90, 0, 0, 39.8, 0.451, 41, 1], [1, 189, 60, 23, 846, 30.1, 0.398, 59, 1], [7, 100, 0, 0, 0, 30.0, 0.484, 32, 1], [8, 125, 96, 0, 0, 0.0, 0.232, 54, 1]]}
[(4.695652173913044, 3.443502215750909), (130.82608695652175, 36.35263173405875), (65.04347826086956, 25.87992950122048), (17.26086956521739, 18.01569364169208), (111.21739130434783, 205.13076905005545), (31.891304347826082, 9.338138467570808), (0.512, 0.4800403392140365), (37.69565217391305, 11.343539541433406)]
<function summarizebyclass at 0x7fc1db452f50>
## Predicted values for test
[0, 0, 1, 0, 1, 0, 1, 0]
## Accuracy
37.5

## I know it's insanely low but that's because we haven't used a lot of data to contribute to probabilities
"""
