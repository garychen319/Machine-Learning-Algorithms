import csv
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

#To run: python3 problem2_3.py input2.csv output2.csv

def main():
	input_f = sys.argv[1]
	output_f = sys.argv[2]
	f = open(input_f)
	csv_f = csv.reader(f)

	#convert to list
	data, years, kilos, meters = [], [], [], []
	for row in csv_f:
		for i in range (0,3):
			row[i] = float(row[i])
		data.append(row)
		years.append(row[0])
		kilos.append(row[1])
		meters.append(row[2])
	n = len(data)

	#scaling
	ymean = np.mean(years);
	kmean = np.mean(kilos);
	ystd = np.std(years);
	kstd = np.std(kilos);

	for i in range (0, len(years)):
		years[i] = (years[i] - ymean)/ystd
	for i in range (0, len(kilos)):
		kilos[i] = (kilos[i] - kmean)/kstd

	results = []

	learn = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5 ,1.0, 5.0, 10.0]
	for a in learn:
		b_0, m_years, m_kilos = 0, 0, 0
		for i in range (0, 100):
			sumB0, sumB1_years, sumB1_kilos = summation(years, kilos, meters, m_years, m_kilos, b_0)
			b_0 = b_0 - (a * sumB0)
			m_years = m_years - (a * sumB1_years)
			m_kilos = m_kilos - (a * sumB1_kilos)
		results.append((a, 100, b_0, m_years, m_kilos))


	#custom input
	a = 0.2
	b_0, m_years, m_kilos = 0, 0, 0
	for i in range (0, 50):
		sumB0, sumB1_years, sumB1_kilos = summation(years, kilos, meters, m_years, m_kilos, b_0)
		b_0 = b_0 - (a * sumB0)
		m_years = m_years - (a * sumB1_years)
		m_kilos = m_kilos - (a * sumB1_kilos)
	results.append((a, 50, b_0, m_years, m_kilos))

	for item in results:
		print(item)

	with open(output_f, 'w') as outcsv:
	    writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
	    for item in results:
	        writer.writerow([item[0], item[1], item[2], item[3], item[4]])	


def summation(years, kilos, meters, m_years, m_kilos, b_0):
	B0, B1_years, B1_kilos = 0.0, 0.0, 0.0
	n = len(years)

	for i in range(0, n):
		fx = function(m_years, years[i], m_kilos, kilos[i], b_0)
		B0 += fx - meters[i]
		B1_years += (fx - meters[i]) * years[i]
		B1_kilos += (fx - meters[i]) * kilos[i]

	return B0/n, B1_years/n, B1_kilos/n

def function(m1, x1, m2, x2, b):
	return (m1*x1) + (m2*x2) + b


# def error(fnc_years, func_kilos, years, kilos, meters):
# 	n = len(years)
# 	sum_years = 0
# 	sum_kilos
# 	for i in range (0, n):
# 		sum_years += (fnc_years(years[i]) - meters[i]) ** 2
# 		sum_kilos += (fnc_kilos(kilos[i]) - meters[i]) ** 2

# 	return (1/(2*n))*sum_years, (1/(2*n))*sum_kilos



# def plot(arr, weights):
# 	#plot points
# 	x = []
# 	y = []
# 	color = []
# 	for row in arr:
# 		x.append(int(row[0]))
# 		y.append(int(row[1]))
# 		color.append(int(row[2]))
# 	plt.scatter(x, y, c = color)

# 	#plot line
# 	i = weights[2] / -weights[1]
# 	s = weights[0] / -weights[1]
# 	ax = plt.axes()
# 	x = np.linspace(0, 10, 1000)
# 	ax.plot(x, i + s*x);

# 	plt.show()

main()













