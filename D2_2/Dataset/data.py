import six.moves.cPickle as pickle
import gzip
import os
import numpy as np
from PIL import Image

def load_data(dataset):
    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:

    return train_set, valid_set, test_set


if __name__ == '__main__':
    train_set, val_set, test_set = load_data('mnist.pkl.gz')

    train_x, train_y = train_set
    val_x, val_y = val_set
    test_x, test_y = test_set
    
    print(train_x.shape)
    print(train_y.shape)
    
    for i in range(100):
        tmp_img = train_x[i].reshape((28,28))*255.9
        samp_img = Image.fromarray(tmp_img.astype(np.uint8))
        samp_img.save('test'+str(i)+'.jpg')
        print(train_y[i])

    mean_img = train_x.mean(1)
    print(train_x)
    print(np.sum(train_x <= 0.1))
    print(np.sum(train_x >= 0.9))
    print(mean_img.shape)

    cov = np.cov(train_x.T)
    print(cov)
    print(cov.shape)