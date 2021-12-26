import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

df = pd.read_csv('titanic-training-dataset-cleaned.csv')

x = df.drop('Survived', axis=1)
y = df['Survived']

# training and evaluating models using Cross-validation
def train_and_evaluate_model(model, name):
    scores = cross_val_score(model, x, y, cv=10, scoring='accuracy')
    print(name + " Cross validated score: " + str(round(scores.mean(),3)))

models = [LogisticRegression(), SVC(), KNeighborsClassifier(n_neighbors = 3), GaussianNB(), Perceptron(), DecisionTreeClassifier(), RandomForestClassifier(n_estimators=100), GradientBoostingClassifier(n_estimators = 100), RidgeClassifierCV()]
model_names = ["Logistic regression", "SVM", "KNN", "Gaussian Naive bayes", "Perceptron", "Decision Tree", "Random Forest", "Gradient boosting", "Ridge classifier CV"]
list(map(train_and_evaluate_model, models, model_names))