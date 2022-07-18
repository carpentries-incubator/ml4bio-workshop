#### Load libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import json
from dtreeviz.trees import dtreeviz

housing_data = pd.read_csv("housing_data.csv")
housing_data = housing_data.drop(['price', 'street','city','statezip'], axis=1)

X = housing_data.drop('classes', axis=1)
y = housing_data['classes']

features = list(X.columns)

tree_model = DecisionTreeClassifier(criterion='gini', max_depth=None, random_state=1)
tree_model = tree_model.fit(X,y)

text_representation = tree.export_text(tree_model, feature_names=features)

with open('decision_tree.json', 'w') as fout:
    fout.write(text_representation)

