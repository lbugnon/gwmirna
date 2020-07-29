# ============================================================
# Leandro Bugnon, lbugnon@sinc.unl.edu.ar
# sinc(i) - http://www.sinc.unl.edu.ar
# ============================================================
# This script run a cross-validation for OC-SVM using CEL dataset.

import numpy as np
import os
import shutil

from random import seed
from sklearn import svm

import time
import warnings

from utils import data_load, get_partitions

warnings.filterwarnings("ignore")

nth = 4  # Set number of threads
out_dir = "results/OC-SVM/"
if not os.path.isdir("results/"):
    os.mkdir("results")
data_dir = 'genomes/'
partitions_dir = "test_partitions/"
nfolds = 8
dataset = "cel"

shutil.rmtree(out_dir, ignore_errors=True)
os.mkdir(out_dir)

verbose = 1

if verbose:
    print("Start...")

# Reproducibility
seed(1)
np.random.seed(1)

filename = "%s%s.csv" % (data_dir, dataset)
print("Loading features...")
data, labels = data_load(filename)
print("Done.")
posind = np.where(labels == 1)[0]  # all known miRNAs

startTime = time.time()

trainTime = []
testTime = []

for fold in range(nfolds):
    print("Training %s with OC-SVM (fold %d of %d)" % (dataset, fold+1, nfolds))
    train_ind, test_ind = get_partitions(partitions_dir, dataset, len(labels), fold)
    startFoldTime = time.time()

    # Train
    toc = time.time()
    classifier = svm.OneClassSVM(kernel="linear")

    classifier.fit(data[train_ind[labels[train_ind] == 1], :])

    trainTime = time.time()-toc

    toc = time.time()
    scores = classifier.decision_function(data[test_ind, :]).squeeze()

    testTime = time.time()-toc
    print("Train %.2f min, Test %.2f min" % (trainTime/60, testTime/60))

    np.savetxt("%s%s_fold%d.csv" % (out_dir, dataset, fold), scores)
