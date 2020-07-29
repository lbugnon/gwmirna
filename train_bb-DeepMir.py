# ============================================================
# Leandro Bugnon, lbugnon@sinc.unl.edu.ar
# sinc(i) - http://www.sinc.unl.edu.ar
# ============================================================
# This script run a cross-validation of the balanced-batch DeepMir classifier. Training method
# was adapted to work with highly imbalanced data and comply  our validation scheme. Original contribution on
# https://github.com/HubertTang/DeepMir

import os
import numpy as np
import pandas as pd
import pickle
import src.bbDeepMir.rna_dataset as rna_dataset
from src.bbDeepMir.rna_model import train
from utils import load_seqs

nfolds = 8
dataset = "cel"
model = "bb-DeepMir"

# Paths
data_dir = "genomes/"
partitions_dir = "test_partitions/"
out_dir = "results/%s/" % model
if not os.path.isdir("results/"):
    os.mkdir("results/")
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

# Load data and labels
print("loading seqs...")
seqs = load_seqs(data_dir, dataset)
print("ok")
labels = pd.read_csv("%s%s.csv" % (data_dir, dataset), usecols=["CLASS"]).values


batch_size = 256
seq_length = 400
n_iter = 20
   
for fold in range(nfolds):

    print("Training %s with %s (fold %d of %d)" % (model, dataset, fold + 1, nfolds))
    test_ind = np.concatenate(
        (np.loadtxt("%smirnas/%s_fold%d.csv" % (partitions_dir, dataset, fold), delimiter=",", dtype=np.int64),
         np.loadtxt("%sunlabeled/%s_fold%d.csv" % (partitions_dir, dataset, fold), delimiter=",", dtype=np.int64)
        ))
    test_ind_set = set(test_ind)
    total_ind = set(np.arange(len(seqs)))
    train_ind = np.array(list(total_ind - test_ind_set))
    
    # generate dataset
    train_generator = rna_dataset.RNA_onehot([seqs[i] for i in train_ind], labels[train_ind], batch_size=batch_size,
                                             dim=(seq_length, 4), num_channels=1, num_classes=2, shuffle=True)

    model = train(train_generator, out_dir, fold, seq_length=seq_length, dataset_name=dataset, n_iter=n_iter)

    test_generator = rna_dataset.RNA_onehot([seqs[i] for i in test_ind], [], batch_size=1,
                                                dim=(seq_length, 4), num_channels=1, num_classes=2, shuffle=False)
    score = model.predict_generator(test_generator)

    np.savetxt("%s%s_fold%d.csv" % (out_dir, dataset, fold), score[:, 1])

