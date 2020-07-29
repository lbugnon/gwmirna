import pandas as pd
import numpy as np

def data_load(filename):
    df = pd.read_csv(filename, sep=",")
    df = df.drop(columns=["sequence_names"])

    labels = df["CLASS"].values.squeeze().astype(int)
    data = df.drop(columns=["CLASS"]).values.astype(float)

    for f in range(data.shape[1]):
        data[np.where(np.isnan(data[:, f]))[0], f] = 0
        data[np.where(np.isinf(data[:, f]))[0], f] = 0


    # Feature Normalization ==========
    normParam = np.zeros((data.shape[1], 2))
    for f in range(data.shape[1]):
        normParam[f, 0] = np.mean(data[:, f])
        normParam[f, 1] = np.std(data[:, f])

        data[:, f] = data[:, f] - normParam[f, 0]
        data[:, f] = data[:, f] / normParam[f, 1]

        data[np.where(np.isnan(data[:, f]))[0], f] = 0
        data[np.where(np.isinf(data[:, f]))[0], f] = 0

    return data, labels


def labels_load(filename):
    df = pd.read_csv(filename, sep=",", usecols=["CLASS"])
    
    labels = df.values.squeeze().astype(int)
   
    return labels


def get_partitions(partitions_dir, dataset, L, fold):
    test_ind = np.concatenate(
        (np.loadtxt("%smirnas/%s_fold%d.csv" % (partitions_dir, dataset, fold), delimiter=",", dtype=np.int64),
         np.loadtxt("%sunlabeled/%s_fold%d.csv" % (partitions_dir, dataset, fold), delimiter=",", dtype=np.int64)
         ))

    train_ind = np.ones(L, dtype=bool)
    full_ind = np.arange(L)
    train_ind[test_ind] = 0
    train_ind = full_ind[train_ind]

    return train_ind, test_ind

def load_seqs(data_dir, dataset):

    df = pd.read_csv("{}{}.csv".format(data_dir, dataset), sep=",", usecols=["sequence_names", "CLASS"])
    labels = df["CLASS"]
    sequences = pd.DataFrame(np.array([df["sequence_names"], np.arange(len(labels))]).T, columns=["seqnames", "ind"]).set_index("seqnames")

    seqnames_fasta = []
    seqs = []
    for label in ["mirnas", "unlabeled"]:
        seq = ""
        for k, line in enumerate(open("%s%s-%s.fasta" % (data_dir, dataset, label))):
            if line[0] == ">":
                if len(seq) > 0:  # save previous seq ind
                    seqs.append(seq)

                seqname = line.strip()[1:]
                seqnames_fasta.append(seqname)
                seq = ""
            else:
                seq = seq + line.strip()
        seqs.append(seq)

    df = pd.DataFrame(np.array([seqnames_fasta, seqs]).T, columns=["seqnames", "seqs"]).set_index("seqnames")

    sequences = df.join(sequences).sort_values(by="ind")
    sequences = sequences[~pd.isna(sequences.ind)].seqs.values

    return sequences


