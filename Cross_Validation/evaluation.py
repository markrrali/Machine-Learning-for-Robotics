#!/usr/bin/env python3

# do not use any other imports!
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.neighbors import KNeighborsClassifier as BlackBoxClassifier
from sklearn.datasets import load_iris

class Evaluation:
    """Class for evaluating classifiers """

    def generate_cv_pairs(self, n_samples, n_folds=5, n_rep=1, rand=False,
                          y=None):
        """ Train and test pairs according to k-fold cross validation

        n_samples (int): The number of samples in the dataset

        n_folds (int, optional (default: 5)): The number of folds for the cross validation

        n_rep (int, optional (default: 1)): The number of repetitions for the cross validation

        rand : boolean, optional (default: False)
            If True the data is randomly assigned to the folds. The order of the
            data is maintained otherwise. Note, *n_rep* > 1 has no effect if
            *random* is False.

        y : array-like, shape (n_samples), optional (default: None)
            If not None, cross validation is performed with stratification and
            y provides the labels of the data.

        Returns
        -------

        cv_splits : list of tuples, each tuple contains two arrays with indices
            The first array corresponds to the training data, the second to the
            testing data for the current split. The list has the length of
            *n_folds* x *n_rep*.

        """
        cv_splits = []
        index = np.arange(0, n_samples) #number of samples in dataset
        subsets_size = n_samples/n_folds #each subset should have a fixed length
        if not random:
            n_rep = 1
        for i in range(n_rep):# perform n_rep times
            if random:
                #randomize the indices
                np.random.shuffle(index)
            if strat:
                index = self.strat(index, n_folds, y)
            initial = 0
            end = subsets_size
            shift = 0

            for j in range(n_folds):
                if j == 0:
                    cv_splits.append((index[int(subsets_size):], index[:int(subsets_size)]))

                elif j == n_folds-1:
                    cv_splits.append((index[:int(initial + shift)], index[int(initial + shift):]))

                else:
                    cv_splits.append((np.concatenate((index[:int(initial + shift)], index[int(end + shift):])),index[int(initial + shift):int(end + shift)]))

                shift += subsets_size

        return cv_splits



    def strat(self, index, n_folds, y):
        """Stratification is a method to have unbiased data, method to eliminate sampling bias in a test set
        It creates a test set with a population that best represents the entire population"""

        """Stratified random sampling is different from simple random sampling,
        which involves the random selection of data from the entire population so that each possible sample is equally likely to occur."""
        ##source: https://medium.com/analytics-vidhya/stratified-sampling-in-machine-learning-f5112b5b9cfe
        indices = np.copy(index) #we dont want to change or affect the real one
        uniq_array, c = np.unique(y, return_counts=True)
        ##returns the array with the unique elements and c is the frequency of each unique elements
        counters = np.copy(c//n_folds)
        #checking the number of each unique member can be placed into each fold (group)
        new = []
        j = 0
        for i in range(n_folds):
            while j<len(indices):
                temp = y[indices[j]] #creating
                if counters[temp]>0:
                    new.append(indices[j])
                    indices = np.delete(indices, j)
                    c[temp] -= 1 #delete 1 counter
                else:
                    j += 1 #move to the next number in array

                if np.sum(counters) == 0: #checking if empty
                    if i != n_folds-1:
                        #if i is not finished and the counter is empty we need to
                        #update what we have left and distribute properly
                        counters = np.copy(counters//(n_folds-i-1))
                    j = 0
                    break
        return new



    def apply_cv(self, X, y, train_test_pairs, classifier):
        """ Use cross validation to evaluate predictions and return performance

        Apply the metric calculation to all test pairs

        Parameters
        ----------

        X : array-like, shape (n_samples, feature_dim)
            All data used within the cross validation

        y : array-like, shape (n-samples)
            The actual labels for the samples

        train_test_pairs : list of tuples, each tuple contains two arrays with indices
            The first array corresponds to the training data, the second to the
            testing data for the current split

        classifier : function
            Function that trains and tests a classifier and returns a
            performance measure. Arguments of the functions are the training
            data, the testing data, the correct labels for the training data,
            and the correct labels for the testing data.

        Returns
        -------

        performance : float
            The average metric value across train-test-pairs
        """
        ### YOUR IMPLEMENTATION GOES HERE ###
        sum = 0
        X_train, Y_train, X_test, Y_test = [], [], [], []
        for i in range(len(train_test_pairs)):
            for j in range(len(train_test_pairs[i][0])+len(train_test_pairs[i][1])):
                if j < len(train_test_pairs[i][0]):
                    X_train.append(X[train_test_pairs[i][0][j]])
                    Y_train.append(y[train_test_pairs[i][0][j]])
                else:
                    X_test.append((X[train_test_pairs[i][1][j-len(train_test_pairs[i][0])]]))
                    Y_test.append((y[train_test_pairs[i][1][j-len(train_test_pairs[i][0])]]))
            sum += classifier(X_train,X_test,Y_train,Y_test)
        ret = sum/len(train_test_pairs)
        return ret

    def black_box_classifier(self, X_train, X_test, y_train, y_test):
        """ Learn a model on the training data and apply it on the testing data

        Parameters
        ----------

        X_train : array-like, shape (n_samples, feature_dim)
            The data used for training

        X_test : array-like, shape (n_samples, feature_dim)
            The data used for testing

        y_train : array-like, shape (n-samples)
            The actual labels for the training data

        y_test : array-like, shape (n-samples)
            The actual labels for the testing data

        Returns
        -------

        accuracy : float
            Accuracy of the model on the testing data
        """
        bbc = BlackBoxClassifier(n_neighbors=10)
        bbc.fit(X_train, y_train)
        acc = bbc.score(X_test, y_test)
        return acc

if __name__ == '__main__':
    # just uncomment the section you would like to run

    #i
    # n_folds = 10
    # n_rep = 1
    # rand = False
    # strat = False

    #ii
    # n_folds = 10
    # n_rep = 10
    # rand = True
    # strat = False

    #iii
    n_folds = 10
    n_rep = 10
    rand = True
    strat = True


    iris = load_iris()

    eval = Evaluation()
    n_samples = len(iris.data)

    cv_pairs = eval.generate_cv_pairs(n_samples, n_folds, n_rep, rand, iris.target)
    Accuracy = eval.apply_cv(iris.data,iris.target,cv_pairs,eval.black_box_classifier)
    print("Accuracy: %s" %Accuracy)
