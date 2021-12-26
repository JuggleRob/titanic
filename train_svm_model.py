import joblib
import pandas as pd
from sklearn.svm import SVC

df = pd.read_csv('titanic-training-dataset-cleaned.csv')

# These features have been chosen with forward sequential feature selection
# For more info: read Walkthrough_the_process Jupyter Notebook
features = ['Pclass', 'Sex', 'AgeGroup', 'FamilySize']
x = df[features]
y = df['Survived']

# Train our model on all of our training data
# SVM has been chosen from a bunch of models
# Using gridsearch we found out that the optimal parameters have the same accuracy result as the default
svm = SVC(kernel='rbf')
svm.fit(x,y)

# Persist our trained model using joblib
joblib.dump(svm, "classifier.joblib")