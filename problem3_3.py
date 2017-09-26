def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#To run: python3 problem3_3.py input3.csv output3.csv
#Program will take ~5 minutes to run

def main():
	input_f = sys.argv[1]
	output_f = sys.argv[2]
	f = open(input_f)
	csv_f = csv.reader(f)

	#convert to list
	data, A, B, Label = [], [], [], []
	for row in csv_f:
		#account for first line "A, B, Label"
		try:
   			val1 = float(row[0])
   			val2 = float(row[1])
   			val3 = int(row[2])
		except ValueError:
   			continue
   		#parse first A,B as floats, Label as int
		for i in range (0,2):
			row[i] = float(row[i])
		row[2] = int(row[2])

		data.append(row)
		A.append(row[0])
		B.append(row[1])
		Label.append(row[2])
	n = len(data)


	# use train/test split with different random_state values
	A_train, A_test, B_train, B_test, L_train, L_test = train_test_split(A, B, Label, train_size = 0.6, test_size = 0.4, stratify = Label)
	AB_train = list(zip(A_train, B_train))
	AB_test = list(zip(A_test, B_test))

	final_results = []

	
	#Linear Kernel
	#Training to find best C
	linear_params = {'C': [0.1, 0.5, 1, 5, 10, 50, 100]}
	linear_grid = GridSearchCV(estimator=SVC(kernel = 'linear'), param_grid=linear_params, cv=5, scoring = 'accuracy')
	linear_grid.fit(AB_train, L_train)

	linear_results = ((linear_grid.best_params_, linear_grid.best_score_))

	#formatting outputted best params from training
	# l1 = []
	# best_params = linear_results[0] #best params is dictionary returned by linear_grid.best_params_
	# l1.append(best_params.get('C'))
	# best_params['C'] = l1

	# #Linear testing with best parameters from training set
	# lin_test_grid = GridSearchCV(estimator=SVC(kernel = 'linear'), param_grid=best_params, cv=5, scoring = 'accuracy')
	# lin_test_grid.fit(AB_test, L_test)
	final_results.append(("svm_linear", linear_results[1], linear_grid.score(AB_test, L_test)))
	
	
	#Polynomial Kernel
	#Training to find best C/degree/gamma
	poly_params = {'C': [0.1, 1, 3], 'degree': [4,5,6], 'gamma': [0.1, 1]}
	poly_grid = GridSearchCV(estimator=SVC(kernel='poly'), param_grid=poly_params, cv=5, scoring = 'accuracy')
	poly_grid.fit(AB_train, L_train)
	poly_results = ((poly_grid.best_params_, poly_grid.best_score_))

	#formatting outputted best params from training
	# l1, l2, l3= [], [], []
	# best_params = poly_results[0] #best params is dictionary returned by poly_grid.best_params_
	# l1.append(best_params.get('C'))
	# l2.append(best_params.get('degree'))
	# l3.append(best_params.get('gamma'))
	# best_params['C'] = l1
	# best_params['degree'] = l2
	# best_params['gamma'] = l3

	# #poly testing with best parameters from training set
	# poly_test_grid = GridSearchCV(estimator=SVC(kernel='poly'), param_grid=best_params, cv=5, scoring = 'accuracy')
	# poly_test_grid.fit(AB_test, L_test)
	final_results.append(("svm_polynomial", poly_results[1], poly_grid.score(AB_test, L_test)))
	

	#RBF Kernel
	#Training to find best C
	RBF_params = {'C': [0.1,0.5,1,5,10,50,100], 'gamma': [0.1,0.5,1,3,6,10]}
	RBF_grid = GridSearchCV(estimator=SVC(kernel='rbf'), param_grid=RBF_params, cv=5, scoring = 'accuracy')
	RBF_grid.fit(AB_train, L_train)

	RBF_results = ((RBF_grid.best_params_, RBF_grid.best_score_))

	# #formatting outputted best params from training
	# l1, l2= [], []
	# best_params = RBF_results[0] #best params is dictionary returned by RBF_grid.best_params_
	# l1.append(best_params.get('C'))
	# l2.append(best_params.get('gamma'))
	# best_params['C'] = l1
	# best_params['gamma'] = l2

	# #RBF testing with best parameters from training set
	# RBF_test_grid = GridSearchCV(estimator=SVC(kernel='rbf'), param_grid=best_params, cv=5, scoring = 'accuracy')
	# RBF_test_grid.fit(AB_test, L_test)
	final_results.append(("svm_rbf", RBF_results[1], RBF_grid.score(AB_test, L_test)))
	

	#Logistic Regression
	#Training to find best C
	logistic_param_grid = {'C': [0.1,0.5,1,5,10,50,100]}
	logistic_grid = GridSearchCV(estimator=LogisticRegression(), param_grid=logistic_param_grid, cv=5, scoring = 'accuracy')
	logistic_grid.fit(AB_train, L_train)

	logistic_results = ((logistic_grid.best_params_, logistic_grid.best_score_))

	# #formatting outputted best params from training
	# l = []
	# best_params = logistic_results[0] #best params is dictionary returned by logistic_grid.best_params_
	# l.append(best_params.get('C'))
	# best_params['C'] = l

	# #Logistic testing with best parameters from training set
	# logistic_test_grid = GridSearchCV(LogisticRegression(), param_grid=best_params, cv=5, scoring = 'accuracy')
	# logistic_test_grid.fit(AB_test, L_test)
	final_results.append(("logistic", logistic_results[1], logistic_grid.score(AB_test, L_test)))
	

	#KNN
	#Training to find best n_neighbors and best leaf
	knn_range = list(range(1,51))
	knn_leaf_size = [5,10,15,20,25,30,35,40,45,50,55,60]
	knn_params = dict(n_neighbors = knn_range, leaf_size = knn_leaf_size)
	knn_grid = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=knn_params, cv=5, scoring = 'accuracy')
	knn_grid.fit(AB_train, L_train)
	knn_results = ((knn_grid.best_params_, knn_grid.best_score_))

	#testing with best parameters obtained from training
	# knn_best_params = knn_results[0]
	# l1, l2 = [], []
	# l1.append(knn_best_params.get('leaf_size'))
	# l2.append(knn_best_params.get('n_neighbors'))
	# knn_best_params['leaf_size'] = l1
	# knn_best_params['n_neighbors'] = l2

	# knn_test_grid = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=knn_best_params, cv=5, scoring = 'accuracy')
	# knn_test_grid.fit(AB_test, L_test)
	final_results.append(("knn", knn_results[1], knn_grid.score(AB_test, L_test)))

	
	#Decision Tree
	#Training to find best n_neighbors and best min_samples_split

	dtree_min_samples_split = [2,3,4,5,6,7,8,9,10]
	dtree_max_depth = list(range(1,51))
	dtree_param_grid = dict(max_depth = dtree_max_depth, min_samples_split = dtree_min_samples_split)
	dtree_grid = GridSearchCV(DecisionTreeClassifier(), dtree_param_grid, cv=5, scoring = 'accuracy')
	dtree_grid.fit(AB_train, L_train)
	dtree_results = ((dtree_grid.best_params_, dtree_grid.best_score_))

	# #testing with best parameters obtained from training
	# params = dtree_results[0]
	# l1, l2 = [], []
	# l1.append(params.get('max_depth'))
	# l2.append(params.get('min_samples_split'))
	# params['max_depth'] = l1
	# params['min_samples_split'] = l2

	# dtree_test_grid = GridSearchCV(DecisionTreeClassifier(), params, cv=5, scoring = 'accuracy')
	# dtree_test_grid.fit(AB_test, L_test)
	final_results.append(("decision_tree", dtree_results[1], dtree_grid.score(AB_test, L_test)))
	

	#Random Forest
	#Training to find best n_neighbors and best min_samples_split
	RF_min_samples_split = [2,3,4,5,6,7,8,9,10]
	RF_max_depth = list(range(1,51))
	RF_param_grid = dict(max_depth = RF_max_depth, min_samples_split = RF_min_samples_split)
	RF_grid = GridSearchCV(RandomForestClassifier(), RF_param_grid, cv=5, scoring = 'accuracy')
	RF_grid.fit(AB_train, L_train)
	RF_results = ((RF_grid.best_params_, RF_grid.best_score_))

	#testing with best parameters obtained from training
	# params = RF_results[0]
	# l1, l2 = [], []
	# l1.append(params.get('max_depth'))
	# l2.append(params.get('min_samples_split'))
	# params['max_depth'] = l1
	# params['min_samples_split'] = l2

	# RF_test_grid = GridSearchCV(RandomForestClassifier(), params, cv=5, scoring = 'accuracy')
	# RF_test_grid.fit(AB_test, L_test)
	final_results.append(("random_forest", RF_results[1], RF_grid.score(AB_test, L_test)))
	

	for elem in final_results:
	 	print(elem)
	
	
	#write to csv file
	with open(output_f, 'w') as outcsv:
	    writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
	    for item in final_results:
	        writer.writerow([item[0], item[1], item[2]])	



main()








