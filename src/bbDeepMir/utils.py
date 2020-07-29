# ============================================================
# Leandro Bugnon, lbugnon@sinc.unl.edu.ar
# sinc(i) - http://www.sinc.unl.edu.ar
# ============================================================
# Adapted from https://github.com/HubertTang/DeepMir
import argparse
import os
import sys
import random
import numpy as np
import pandas as pd



def cmd_onehot():
    parser = argparse.ArgumentParser(description="ARGUMENTS")

    # argument for dataset
    parser.add_argument(
        'dataset',
        type=str,
        help="Dataset directory (MUST in `inputdir` directory)"
    )

    parser.add_argument(
        "num_classes",
        type=int,
        help="Number of families")

    parser.add_argument(
        "--seq_length",
        default=200,
        type=int,
        help="Length of RNA sequence (default: 200)")

    # argument for training
    # parser.add_argument(
    #     '--model',
    #     type=str,
    #     default='DeepRfam',
    #     help="Choose model (DeepRfam(default), DeepRfam_deep, DeepRfam_lenet, ImgFam, L4Fam, L4BNFam, L5CFam, L5Fam, "
    #          "L5CFam_nopooling, L5CFam_dilation, L5CFam_ave, L5CFam_temp, L6Fam, L7CFam, Github_scnn)"
    # )

    # parser.add_argument(
    #     '--encode',
    #     type=str,
    #     default='RNA_onehot',
    #     help="Choose encoding method (RNA_onehot(default), RNA_img, RNA_fimg, RNA_pimg)"
    # )

    parser.add_argument(
        '--learning_rate',
        default=0.001,
        type=float,
        help='Initial learning rate (default: 0.001)'
    )

    parser.add_argument(
        '--batch_size',
        default=64,
        type=int,
        help='Batch size (default: 64)'
    )

    parser.add_argument(
        '--num_epochs',
        default=5,
        type=int,
        help="Number of epochs to train (default: 5)"
    )

    # parser.add_argument(
    #     '--train_file',
    #     type=str,
    #     default=f'train.csv',
    #     help="Input data for training (default: train.csv)"
    # )
    #
    # parser.add_argument(
    #     '--valid_file',
    #     type=str,
    #     default=f'validation.csv',
    #     help="Input data for validation (default: validation.csv)"
    # )
    #
    # parser.add_argument(
    #     '--test_file',
    #     type=str,
    #     default='test.csv',
    #     help="Input data for training (default: test.csv)"
    # )
    #
    # parser.add_argument(
    #     '--family_dict_file',
    #     type=str,
    #     default='fam_label.csv',
    #     help="Input data for training (default: fam_label.csv)"
    # )
    #
    parser.add_argument(
        '--filter_sizes',
        default=[2, 4, 6, 8, 10, 12, 14, 16],
        type=int,
        nargs='+',
        help="Space separated list of motif filter lengths. (ex, --filter_sizes 4 8 12)\
            \n(default: [2, 4, 6, 8, 10, 12, 14, 16])"
    )

    parser.add_argument(
        '--num_filters',
        default=256,
        type=int,
        help='Number of filters per kernel (default: 256)'
    )

    parser.add_argument(
        '--keep_prob',
        type=float,
        default=0.7,
        help='Rate to be kept for dropout. (default: 0.7)'
    )

    parser.add_argument(
        '--num_hidden',
        type=int,
        default=512,
        help='Number of neurons in hidden layer. (default: 512)'
    )

    parser.add_argument(
        '--loss_function',
        type=str,
        default="categorical_crossentropy",
        help='Loss function. (default: categorical_crossentropy)'
    )

    parser.add_argument(
        '--optimizer',
        type=str,
        default="Adam",
        help='Optimizer. (default: Adam)'
    )

    # for logging
    parser.add_argument(
        '--log_name',
        type=str,
        default=None,
        help="Name for logging. (default: local_time)"
    )

    parser.add_argument(
        '--log_dir',
        type=str,
        default="log",
        help="Directory for logging. (default: log)"
    )

    # parser.add_argument(
    #     '--remark',
    #     type=str,
    #     default=None,
    #     help="Remark additional information"
    # )

    # version
    parser.add_argument(
        '-v', '--version',
        action='version',
        version='version_2.0'
    )

    args = parser.parse_args()

    # assert (args.model in ['DeepRfam', 'L5CFam'])
    # assert (args.encode.split('#').pop(0) in ['RNA_onehot', 'RNA_img'])

    return args


def cmd_img():
    parser = argparse.ArgumentParser(description="ARGUMENTS")

    # argument for dataset
    parser.add_argument(
        'dataset',
        type=str,
        help="Dataset directory (MUST in `inputdir` directory)"
    )

    parser.add_argument(
        "num_classes",
        type=int,
        help="Number of families")

    parser.add_argument(
        "--seq_length",
        default=200,
        type=int,
        help="Length of RNA sequence (default: 200)")

    # argument for training
    # parser.add_argument(
    #     '--model',
    #     type=str,
    #     default='DeepRfam',
    #     help="Choose model (DeepRfam(default), DeepRfam_deep, DeepRfam_lenet, ImgFam, L4Fam, L4BNFam, L5CFam, L5Fam, "
    #          "L5CFam_nopooling, L5CFam_dilation, L5CFam_ave, L5CFam_temp, L6Fam, L7CFam, Github_scnn)"
    # )

    # parser.add_argument(
    #     '--encode',
    #     type=str,
    #     default='RNA_onehot',
    #     help="Choose encoding method (RNA_onehot(default), RNA_img, RNA_fimg, RNA_pimg)"
    # )

    parser.add_argument(
        '--learning_rate',
        default=0.001,
        type=float,
        help='Initial learning rate (default: 0.001)'
    )

    parser.add_argument(
        '--batch_size',
        default=32,
        type=int,
        help='Batch size (default: 32)'
    )

    parser.add_argument(
        '--num_epochs',
        default=5,
        type=int,
        help="Number of epochs to train (default: 5)"
    )

    parser.add_argument(
        '--filter_sizes',
        default=[2, 2],
        type=int,
        nargs='+',
        help="Space separated list of motif filter lengths. (e.g., --filter_sizes 3 5) (default: [2, 2])"
    )

    parser.add_argument(
        '--num_filters',
        default=[32, 64],
        type=int,
        nargs='+',
        help='Number of filters per kernel in two convolution layers. (e.g., --num_filters 16 32) (default: [32, 64])'
    )

    parser.add_argument(
        '--keep_prob',
        type=float,
        default=0.5,
        help='Rate to be kept for dropout. (default: 0.7)'
    )

    parser.add_argument(
        '--num_hidden',
        type=int,
        default=[128, 64],
        help='Number of neurons in first two hidden layers. (e.g., --num_hidden 64 32) (default: [128, 64])'
    )

    parser.add_argument(
        '--loss_function',
        type=str,
        default="categorical_crossentropy",
        help='Loss function. (default: categorical_crossentropy)'
    )

    parser.add_argument(
        '--optimizer',
        type=str,
        default="Adam",
        help='Optimizer. (default: Adam)'
    )

    # for logging
    parser.add_argument(
        '--log_name',
        type=str,
        default=None,
        help="Name for logging. (default: local_time)"
    )

    parser.add_argument(
        '--log_dir',
        type=str,
        default="log",
        help="Directory for logging. (default: log)"
    )

    # parser.add_argument(
    #     '--remark',
    #     type=str,
    #     default=None,
    #     help="Remark additional information"
    # )

    # version
    parser.add_argument(
        '-v', '--version',
        action='version',
        version='version_2.0'
    )

    args = parser.parse_args()

    # assert (args.model in ['DeepRfam', 'L5CFam'])
    # assert (args.encode.split('#').pop(0) in ['RNA_onehot', 'RNA_img'])

    return args


def count_lines(file_name):
    with open(file_name) as f:
        line_count = sum(1 for line in f if line.strip())
    return line_count





def k_fold_split_csv(dir_name, filename='train.csv', k=10):
    """Split the origin csv file into k fold csv file.
    """
    # initialization
    seq_list = []
    k_list = []

    with open(f"data/{dir_name}/{filename}", 'r') as f:
        for line in f:
            line = line.strip()
            seq_list.append(line)

    random.shuffle(seq_list)

    num_each = int(len(seq_list) / k)

    # generate a list containing k_fold dataset
    for i in range(k):
        temp_list = seq_list[num_each * i:num_each * (i + 1)]
        k_list.append(temp_list)

    for i in range(k):

        test_list = k_list.pop(0)

        for test_seq in test_list:
            with open(f"data/{dir_name}/test_{i}.csv", 'a') as f:
                print(test_seq, file=f)

        for train_sub_list in k_list:
            for train_seq in train_sub_list:
                with open(f"data/{dir_name}/train_{i}.csv", 'a') as f:
                    print(train_seq, file=f)

        k_list.append(test_list)


if __name__ == "__main__":
    cmd()
