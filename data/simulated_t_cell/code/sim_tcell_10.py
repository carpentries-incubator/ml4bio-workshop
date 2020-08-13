import numpy as np
import pandas as pd
import random
from sklearn.datasets import make_blobs

data = make_blobs(n_samples = 500, n_features = 2, centers = 2, random_state=0)
X, y = data
X = pd.DataFrame(X, columns=['cell_size','total_intensity'])
y = pd.DataFrame(y, columns=['class'])
data = pd.concat([X, y], axis=1)
data['class'] = data['class'].replace(0,'quiescent')
data['class'] = data['class'].replace(1,'activated')

def random_generator(lst, start, end):
    random.seed(1)
    for i in range(500):
        n = round(random.uniform(start, end),2)
        lst.append(n)
    return lst

feature_a = []
a = random_generator(feature_a, 10, 24)
feature_b = []
b = random_generator(feature_b, 50, 100)
feature_c = []
c = random_generator(feature_c, 10,24)
feature_d = []
d = random_generator(feature_d, 10,24)
feature_e = []
e = random_generator(feature_e, 0, 500)
feature_f = []
f = random_generator(feature_f, 1, 10)
feature_g = []
g = random_generator(feature_g, 1, 20)
feature_h = []
h = random_generator(feature_h, 1, 20)

#data.drop(columns = ['donor'], axis = 1, inplace = True)

data.insert(2,'feature_a', a)
data.insert(3,'feature_b', b)
data.insert(4,'feature_c', c)
data.insert(5,'feature_d', d)
data.insert(6,'feature_e', e)
data.insert(7,'feature_f', f)
data.insert(8,'feature_g', g)
data.insert(9,'feature_h', h)

data.to_csv('../simulated_t_cells_10.csv', sep=' ', index=False)
