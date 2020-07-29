# ============================================================
# Leandro Bugnon, lbugnon@sinc.unl.edu.ar
# sinc(i) - http://www.sinc.unl.edu.ar
# ============================================================
# This script run a cross-validation for deeSOM using CEL dataset.

import numpy as np
import os
import shutil
from random import seed
import time
import warnings
from deesom import DeeSOM
import pickle
from utils import data_load, get_partitions

warnings.filterwarnings("ignore")

nth = 4  # Set number of threads
out_dir = "results/deeSOM/"
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
posind = np.where(labels == 1)[0]  # all known miRNAs
print("Done.")
startTime = time.time()

trainTime = []
testTime = []

for fold in range(nfolds):
    print("Training deeSOM with %s (fold %d of %d)" % (dataset, fold+1, nfolds))
    train_ind, test_ind = get_partitions(partitions_dir, dataset, len(labels), fold)

    startFoldTime = time.time()

    # Train
    labels[posind] = 1  # Starting with all known genome-wide positives...
    print("original pos %d" % sum(labels))
    # ...remove the positive labels that will be used for testing
    labels[test_ind] = 0
    print("train pos %d" % sum(labels))

    toc = time.time()
    # Train
    deesom = DeeSOM(verbosity=1)
    deesom.fit(data, labels)

    trainTime = time.time()-toc
    # Save the scores for the test partition.

    toc = time.time()
    scores = deesom.predict_proba()[test_ind]

    testTime = time.time()-toc
    print("Train %.2f min, Test %.2f min" % (trainTime/60, testTime/60))

    np.savetxt("%s%s_fold%d.csv" % (out_dir, dataset, fold), scores)

