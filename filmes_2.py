# -*- coding: utf-8 -*-
"""
Created on Fri May 24 22:58:43 2019

@author: taciusm
"""

import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neural_network import MLPClassifier

if __name__ == "__main__":
    # NOTE: we put the following in a 'if __name__ == "__main__"' protected
    # block to be able to use a multi-core grid search that also works under
    # Windows, see: http://docs.python.org/library/multiprocessing.html#windows
    # The multiprocessing module is used as the backend of joblib.Parallel
    # that is used when n_jobs != 1 in GridSearchCV

    # the training data folder must be passed as first argument
    movie_reviews_data_folder = r"C:\Users\taciusm\Downloads\data"
    dataset = load_files(movie_reviews_data_folder, shuffle=False)
    print("n_samples: %d" % len(dataset.data))

    # split the dataset in training and test set:
    docs_train, docs_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.3, random_state=None)

    # TASK: Build a vectorizer / classifier pipeline that filters out tokens
    # that are too rare or too frequent
    svc = Pipeline([('vect' , TfidfVectorizer(analyzer='word', min_df=3, max_df=0.95)), ('ann',MLPClassifier(activation = 'tanh'))])
    

    # TASK: Build a grid search to find out whether unigrams or bigrams are
    # more useful.
    params ={'vect__ngram_range':[(1,1),(2,2)]}
    ann = GridSearchCV(estimator=svc, param_grid= params)
    
    # Fit the pipeline on the training set using grid search for the parameters
    ann.fit(docs_train, y_train)
    # TASK: print the cross-validated scores for the each parameters set
    # explored by the grid search
    print(ann.grid_scores_)
    
    # TASK: Predict the outcome on the testing set and store it in a variable
    # named y_predicted
    y_predicted = ann.predict(docs_test)
    # Print the classification report
    print(metrics.classification_report(y_test, y_predicted,
                                        target_names=dataset.target_names))

    # Print and plot the confusion matrix
    cm = metrics.confusion_matrix(y_test, y_predicted)
    print(cm)

    #import matplotlib.pyplot as plt
    #plt.matshow(cm)
    #plt.show()
