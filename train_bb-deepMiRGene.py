# ============================================================
# Leandro Bugnon, lbugnon@sinc.unl.edu.ar
# sinc(i) - http://www.sinc.unl.edu.ar
# ============================================================
# This script run a cross-validation of the balanced-batch deepMiRGene classifier. Training method
# was adapted to work with highly imbalanced data and comply  our validation scheme. Original contribution on
# https://github.com/eleventh83/deepMiRGene

import os
import pickle
import shutil
import pandas as pd
import readline

readline.parse_and_bind('tab:complete')
import numpy as np
import time
from keras.utils import to_categorical
from utils import load_seqs

from src.bbdeepMiRGene.deepmirgene import SoftAttention, mymodel, import_data, one_hot_wrap



MAX_LEN = 400 # maximum sequence length
DIM_ENC = 16 # dimension of a one-hot encoded vector (e.g., 4 (sequence) x 4 (structure) = 16)
DIM_LSTM1 = 20
DIM_LSTM2 = 10
DIM_DENSE1 = 400
DMI_DENSE2 = 100
N = 30000#150000 # number of sequences to process at a time (adjust according to available RAM)

N_EPOCH = 30 


model = "bb-deepMiRGene"
data_dir = 'genomes/'
partitions_dir = "test_partitions/"
nfolds = 8
out_dir = "results/%s/" % model
dataset = "cel"

if not os.path.isdir("results/"):
    os.mkdir("results")
if not os.path.isdir("tmp/"):
    os.mkdir("tmp/")



shutil.rmtree(out_dir, ignore_errors=True)
os.mkdir(out_dir)

verbose = 1

labels = pd.read_csv("%s%s.csv" % (data_dir, dataset)).CLASS.values

tstart = time.time()
print("Loading and parsing seqs (it may take several minutes)...")
data = import_data(load_seqs(data_dir, dataset))
print("Done")


for fold in range(nfolds):

    print("Training %s with %s (fold %d of %d)" % (model, dataset, fold+1, nfolds))
    test_ind = np.concatenate((np.loadtxt("%smirnas/%s_fold%d.csv" %(partitions_dir, dataset, fold), delimiter=",", dtype=np.int64),
                               np.loadtxt("%sunlabeled/%s_fold%d.csv" % (partitions_dir, dataset, fold), delimiter=",", dtype=np.int64)
    ))
    train_ind = np.array([n for n in np.arange(len(data)) if n not in test_ind])

    # preprocess and save tensor for training
    # pos samples
    X = [data[t] for t in train_ind[labels[train_ind] == 1]]
    X_pos = one_hot_wrap(X, MAX_LEN, DIM_ENC)
    Y_pos = to_categorical([1]*len(X_pos), num_classes=2)
    npos = X_pos.shape[0]

    X = [data[t] for t in train_ind[labels[train_ind] == 0]]
    model = mymodel()
    for epoch in range(N_EPOCH):
        # neg samples
        k = 0
        while k*N <= len(X):
            print("train encoding %d part - epoch %d" % (k, epoch))
            N_sub = len(X[k*N:(k+1) * N])
            X_sub = np.zeros((npos + N_sub, MAX_LEN, DIM_ENC))
            X_sub[:npos, :, :] = X_pos
            Y_sub = np.zeros((npos+N_sub, 2))
            Y_sub[:npos, :] = Y_pos
            X_sub[npos:, :, :] = one_hot_wrap(X[k*N:(k+1) * N], MAX_LEN, DIM_ENC)
            Y_sub[npos:, :] = to_categorical([0]*(X_sub.shape[0]-npos), num_classes=2)
            history = model.fit(X_sub, Y_sub, epochs=1, verbose=1, batch_size=512, class_weight='auto')
            k += 1

    X = [data[t] for t in test_ind]

    predictions = []
    k = 0
    while k*N <= len(X):
        print("test: encoding x part", k)
        X_part = one_hot_wrap(X[k*N:(k+1) * N], MAX_LEN, DIM_ENC)
        predictions.append(model.predict(X_part, verbose=1))
        k += 1

    np.savetxt("%s%s_fold%d.csv" % (out_dir, dataset, fold), np.concatenate(predictions)[:, 1])



