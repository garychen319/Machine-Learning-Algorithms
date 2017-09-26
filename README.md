# Machine-Learning-Algorithms


I. Perceptron Learning Algorithm

II. Linear Regression

III. Classification

--------------------------------------------
I. Perceptron

$ python3 problem1_3.py input1.csv output1.csv

This should generate an output file called output1.csv. With each iteration of your PLA, the program will print a new line to the output file, containing a comma-separated list of the weights w_1, w_2, and b in that order.
Upon convergence, the program will stop, and the final values of w_1, w_2, and b will be printed to the output file. This defines the decision boundary that your PLA has computed for the given dataset.

II. Linear Regression

$ python3 problem2_3.py input2.csv output2.csv

This should generate an output file called output2.csv. There are ten cases in total, nine with the specified learning rates (and 100 iterations), and one with an arbitrary choice of learning rate.
After each of these ten runs, the program will print a new line to the output file, containing a comma-separated list of alpha, number_of_iterations, b_0, b_age, and b_weight in that order. These represent the regression models that your gradient descents have computed for the given dataset.

III. Classification

• SVM with Linear Kernel. Observed the performance of the SVM with linear kernel and searched for a good setting of parameters to obtain high classification accuracy. Specifically, values of C = [0.1, 0.5, 1, 5, 10, 50, 100]. 

• SVM with Polynomial Kernel.
Try values of C = [0.1, 1, 3], degree = [4, 5, 6], and gamma = [0.1, 1].

• SVM with RBF Kernel.
Try values of C = [0.1, 0.5, 1, 5, 10, 50, 100] and gamma = [0.1, 0.5, 1, 3, 6, 10].

• Logistic Regression. Try values of C = [0.1, 0.5, 1, 5, 10, 50, 100].

• k-Nearest Neighbors.
Try values of n_neighbors = [1, 2, 3, ..., 50] and leaf_size = [5, 10, 15, ..., 60].

• Decision Trees.
Try values of max_depth = [1, 2, 3, ..., 50] and min_samples_split = [1, 2, 3, ..., 10].

• Random Forest.
Try values of max_depth = [1, 2, 3, ..., 50] and min_samples_split = [1, 2, 3, ..., 10].

The file will contain an entry for each of the seven methods used. For each method, output will be a comma-separated list, including the method name, best score, and test score.
