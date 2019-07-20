# Import packages
import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.ensemble import RandomForestClassifier

# Plotting packages
from matplotlib import rcParams
rcParams['xtick.major.pad'] = 1
rcParams['ytick.major.pad'] = 1


class IterML(object):

    def __init__(self, project):
        self.project = project
        self.models = self.create_model_list()


    def create_model_list(self):
        models = []
        models.append(('LR', LogisticRegression()))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('DTC', DecisionTreeClassifier()))
        models.append(('RFC', RandomForestClassifier()))
        models.append(('SVM', SVC()))
        models.append(('NB', MultinomialNB()))
        return models

    def iter_models(self, X_train, X_test, y_train, y_test):

        results = []
        names = []
        scoring = 'accuracy'
        for name, model in models:
            kfold = KFold(len(X_train), n_folds=5, random_state=2, shuffle=True)
            cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            mod = model
            mod.fit(X_train, y_train)
            mod_pred = mod.predict(X_test)
            mod_accuracy = accuracy_score(mod_pred, y_test)
            mod_rmse = (mse(mod_pred, y_test) ** 1 / 2)
            msg = "CV Accuracy %s: %f SD %f - Test Accuracy: %f RMSE: %f" % (
            name, cv_results.mean(), cv_results.std(), mod_accuracy, mod_rmse)
            print(msg)

    def iter_models_grid(self, X_train, y_train):
        # Define parameters optimization with GridSearchCV.
        SVM_params = {'C': [0.1, 10, 100], 'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}
        LR_params = {'C': [0.001, 0.1, 1, 10, 100]}
        LDA_params = {'n_components': [None, 1, 2, 3], }
        KNN_params = {'n_neighbors': [1, 5, 10, 20], 'p': [2]}
        RF_params = {'n_estimators': [10, 50, 100]}
        DTC_params = {'criterion': ['entropy', 'gini'], 'max_depth': [10, 50, 100]}
        NB_mult_params = {'alpha': [1, 10]}

        # Make list of models to test with paramter dictionaries.
        models_opt = []
        models_opt.append(('LR', LogisticRegression(), LR_params))
        models_opt.append(('LDA', LinearDiscriminantAnalysis(), LDA_params))
        models_opt.append(('KNN', KNeighborsClassifier(), KNN_params))
        models_opt.append(('DTC', DecisionTreeClassifier(), DTC_params))
        models_opt.append(('RFC', RandomForestClassifier(), RF_params))
        models_opt.append(('SVM', SVC(), SVM_params))
        models_opt.append(('NB', MultinomialNB(), NB_mult_params))

        results_params = []
        names_params = []
        scoring = 'accuracy'

        for name, model, params in models_opt:
            kfold = KFold(len(X_train), n_folds=5, random_state=2, shuffle=True)
            model_grid = GridSearchCV(model, params)
            cv_results_params = cross_val_score(model_grid, X_train, y_train, cv=kfold, scoring=scoring)
            results_params.append(cv_results_params)
            names_params.append(name)
            msg = "CV Accuracy %s: %f (%f)" % (name, cv_results_params.mean(), cv_results_params.std())
            print(msg)


