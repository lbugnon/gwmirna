# ============================================================
# Leandro Bugnon, lbugnon@sinc.unl.edu.ar
# sinc(i) - http://www.sinc.unl.edu.ar
# ============================================================
# Adapted from https://github.com/HubertTang/DeepMir

import numpy as np
import pandas as pd
import keras
from keras.utils import to_categorical
import itertools
import math

RNASET = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3,
          'N': 'n', 'Y': 'y', 'R': 'r', 'W': 'w',
          'M': 'm', 'K': 'k', 'S': 's', 'B': 'b',
          'H': 'h', 'D': 'd', 'V': 'v', 'X': 'x'
          }

# =========================== one_hot_encoding ========================================================
def encoding_1hot(seq, seq_length=312):
    arr = np.zeros((1, seq_length, 4, 1))
    for i, c in enumerate(seq):
        if i < seq_length:
            if type(RNASET[c]) == int:
                idx = RNASET[c]
                arr[0][i][idx][0] = 1
            else:
                continue

    return arr


def RNAonehot_generator(file, batch_size=64, num_classes=142, seq_length=312, shuffle=True):
    """Using python generator to generate batch of dataset
    """

    df = pd.read_csv(file, sep=',', header=None)
    indexs = [tmp for tmp in range(len(df[0]))]

    while True:
        # select sequences for the csv file
        if shuffle:
            np.random.shuffle(indexs)
        for i in range(0, len(indexs), batch_size):
            ids = indexs[i:i + batch_size]

            seq_np = np.zeros((len(ids), seq_length, 4, 1), dtype=np.float32)
            labels = np.zeros((len(ids), num_classes))

            for n in range(len(ids)):
                seq_np[n] = encoding_1hot(df[1][ids[n]])
                labels[n] = to_categorical(df[0][ids[n]], num_classes)

            yield (seq_np, labels)


class RNA_onehot(keras.utils.Sequence):
    """Generates data for Keras. https://github.com/afshinea/keras-data-generator/blob/master/my_classes.py
    """

    def __init__(self, seqs, labels, batch_size=64, dim=(400, 4), num_channels=1,
                 num_classes=143, shuffle=True):
        """Initialization
        """
        self.seqs = [s.upper() for s in seqs] # Convert to uppercase
        self.labels = labels
        self.pos_ind = np.where(self.labels==1)[0]
        self.neg_ind = np.where(self.labels==0)[0]
        
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = num_channels
        self.n_classes = num_classes
        self.shuffle = False#shuffle
        self.n_iter = 10
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        """
        if self.batch_size == 1:
            return len(self.seqs)
        else:
            return self.n_iter
    
    def __getitem__(self, index):
        """Generate one batch of data
        """
        # Generate indexes of the batch

        if self.batch_size>1:
            indexes = np.concatenate((np.random.choice(self.pos_ind, size=self.batch_size//10, replace=False), np.random.choice(self.neg_ind, size=self.batch_size - self.batch_size//10, replace=False) ), axis=0)
        else:
            indexes = np.array([self.indexes[index]])
        
        # Generate data
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.seqs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        """Generates data containing batch_size samples
        """  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        if len(self.labels)>0:
            labels = self.labels[indexes]
        else:
            labels = [-1]
        seqs = [self.seqs[s] for s in indexes]

        # Generate data
        for i, s in enumerate(seqs):
            # Store sample
            X[i, ] = encoding_1hot(s, seq_length=self.dim[0])
            # Store class
            y[i] = labels[i]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


