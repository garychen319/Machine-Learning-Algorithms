import csv
import sys
import numpy as np
import matplotlib.pyplot as plt

#To run: python3 problem1_3.py input1.csv output1.csv

def main():
	input_f = sys.argv[1]
	output_f = sys.argv[2]
	f = open(input_f)
	csv_f = csv.reader(f)

	#convert to list
	arr = []
	vals = [] # actual 1/-1 values of each row
	for row in csv_f:
		for i in range (0,3):
			row[i] = int(row[i])
		arr.append(row)
		vals.append(row[2])

	#variable initialization
	weights = [0, 0, 0]
	n = len(arr)
	predictions = [0] * n
	output = []
	
	#Perceptron Algorithm
	while(checkError(predictions, vals) == 0): # until no error
		for i in range (0, n): # for each row
			row = arr[i]
			yi = row[2] #1/-1 val
			f = function(row, weights)
			predictions[i] = f
			if (yi * f) <= 0: #misprediction
				weights[0] = weights[0] + (yi * row[0])
				weights[1] = weights[1] + (yi * row[1])
				weights[2] = weights[2] + (row[2])
		#print(weights)
		output.append([weights[0], weights[1], weights[2]])

	#plot(arr, weights)
	
	#write to csv file
	with open(output_f, 'w') as outcsv:
	    writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
	    for item in output:
	        writer.writerow([item[0], item[1], item[2]])

def function(row, weights):
	s = ((weights[0] * row[0]) + (weights[1] * row[1]) + weights[2])
	if s > 0:
		return 1
	else:
		return -1

def checkError(predictions, vals):
	for i in range (1, len(vals)):
		if(predictions[i] != vals[i]):
			return 0 #error
	return 1 #no error


def plot(arr, weights):
	#plot points
	x = []
	y = []
	color = []
	for row in arr:
		x.append(int(row[0]))
		y.append(int(row[1]))
		color.append(int(row[2]))
	plt.scatter(x, y, c = color)

	#plot line
	i = weights[2] / -weights[1]
	s = weights[0] / -weights[1]
	ax = plt.axes()
	x = np.linspace(0, 10, 1000)
	ax.plot(x, i + s*x);

	plt.show()

main()













