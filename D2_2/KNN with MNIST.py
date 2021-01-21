import six.moves.cPickle as pickle
import gzip
import os
import numpy as np
from PIL import Image
import matplotlib.pylab as plt
import math
import torch

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

def Image_show(n):
    image = test_image[n]
    label = test_label[n]
    image = image.reshape(28, 28)
    image.shape

    plt.figure(figsize=(4, 4))
    plt.title("answer: " + str(label))
    plt.imshow(image, cmap="gray")
    plt.show()

def Euclidean_Distance(test, train): #Euclidean Distance
    result = test - train
    result = np.square(result)

    result = result.sum()
    result = math.sqrt(result)
    return result

def Manhattan_Distance(test, train):
    result = test - train
    result = np.abs(result)
    result = result.sum()
    return result


def DefineClass(test, k, batch_size):
    distance = []
    n_list = [0*i for i in range(10)]

    batch = np.random.randint(0, TRAIN_LEN, size = batch_size)

    for i in batch:
        distance.append([Euclidean_Distance(test, train_image[i]), train_label[i]])
    distance.sort()

    for i in range(k):
        n_list[distance[k][1]] +=1

    max_i = 0
    for i in range(10):
        if(n_list[i] >= n_list[max_i]):
            max_i = i

    return max_i

def FindAccuracy(k, batch_size):
    total = 0
    for i in range(TEST_LEN):
        if DefineClass(test_image[i], k, batch_size) == test_label[i]:
            total +=1
        if(i % 500 == 0):
            print(i/TEST_LEN * 100, "%", end=" ")

    accuracy = total / TEST_LEN * 100
    return accuracy

print("정확도: ",FindAccuracy(5, 1000), "%")