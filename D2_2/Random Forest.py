import numpy as np
import matplotlib.pyplot as plt
import six.moves.cPickle as pickle
import gzip
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

LENGTH = 784
TRAIN_LEN = 50000
TEST_LEN = 10000

with gzip.open('Dataset\mnist.pkl.gz', 'rb') as f:
    try: 
        train_set, vaild_set, test_set = pickle.load(f, encoding='latin1')
    except:
        print("GZIP Error")
        exit()

train_image, train_label = train_set
val_image, val_label = vaild_set
test_image, test_label = test_set

rf = RandomForestClassifier(random_state = 0)
rf.fit(train_image, train_label)

pred = rf.predict(test_image)
print("Accuracy: ", accuracy_score(test_label, pred)*100, '%')