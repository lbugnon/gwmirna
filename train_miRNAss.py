# ============================================================
# Leandro Bugnon, lbugnon@sinc.unl.edu.ar
# sinc(i) - http://www.sinc.unl.edu.ar
# ============================================================
# This script run a cross-validation for mirRNAss using the CEL dataset.
# miRNAss is distributed in a R package in https://cran.r-project.org/web/packages/miRNAss/index.html

import numpy as np
import os
import shutil

from random import seed

import time
import warnings
import pickle
import pandas as pd

from utils import data_load, labels_load, get_partitions

warnings.filterwarnings("ignore")

out_dir = "results/miRNAss/"
data_dir = 'genomes/'
partitions_dir = "test_partitions/"
nfolds = 8
dataset = "cel"

if not os.path.isdir("results/"):
    os.mkdir("results/")
if not os.path.isdir("tmp/"):
    os.mkdir("tmp/")

shutil.rmtree(out_dir, ignore_errors=True)
os.mkdir(out_dir)

verbose = 1

if verbose:
    print("Start...")

# Reproducibility
seed(1)
np.random.seed(1)

filename = "%s%s.csv" % (data_dir, dataset)
print("Preparing labels")
labels = labels_load(filename)
print("Done.")
posind = np.where(labels == 1)[0]  # all known miRNAs

startTime = time.time()

trainTime = []
testTime = []

for fold in range(nfolds):
    print("Training %s with miRNAss (fold %d of %d)" % (dataset, fold+1, nfolds))
    train_ind, test_ind = get_partitions(partitions_dir, dataset, len(labels), fold)
    startFoldTime = time.time()

    # Save labels for train (1 well-known mirnas, -1 negatives, 0 labels to predict)
    indexes = np.zeros(len(train_ind)+len(test_ind))
    indexes[train_ind[labels[train_ind] == 1]] = 1
    indexes[train_ind[labels[train_ind] == 0]] = -1
    np.savetxt("tmp/labels_mirnass.csv", indexes, delimiter=",")

    # run miRNAss
    command = "R  < src/miRNAss/train_fold.R %s %s --no-save" % (data_dir, dataset) 
    print("running '%s'..." % command)
    os.system(command)

    # save scores
    
    scores_all = pd.read_csv("tmp/prediction_mirnass.csv")["V1"].values
    scores_test = scores_all[test_ind]
    

    np.savetxt("%s%s_fold%d.csv" % (out_dir, dataset, fold), scores_test)
    
