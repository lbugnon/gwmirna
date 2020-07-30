# Genome-wide discovery of pre-miRNAs

This repository contains the source code to reproduce the results shown in

> L.A. Bugnon, C. Yones, D.H. Milone and G. Stegmayer, "Genome-wide discovery of pre-miRNAs: comparison of recent approaches based on machine learning", Briefings in Bioinformatics, (in press), 2020

The genome-wide discovery of microRNAs (miRNAs) involves identifying sequences having the highest chance of being a novel  miRNA precursor (pre-miRNA), within all  sequences in a complete genome. The known pre-miRNAs are usually just a few in comparison to the millions of candidates to pre-miRNAs that have to be analyzed. In this work, we review  six recent methods for tackling this problem with machine learning, comparing the models in genome-wide datasets. They have been designed for the pre-miRNAs prediction task, where there is a class of interest that is significantly underrepresented (the known pre-miRNAs) with respect to a very large number of unlabeled samples. 

## Running the experiments

The methods reviewed here were published under different frameworks. Several installation steps would be needed for each case. Most of the methods run on Python, except for miRNAss that runs on R and deepBN that runs on Matlab/Octave. The code was tested with:
- [Python 3.7](https://www.python.org/downloads/)
- [Octave 4.2](https://www.gnu.org/software/octave/#install)
- [R 3.5.2](https://www.r-project.org/)

Unzip the source code from "gwmirna-X.Y.tar.gz". Sequences and features to be analyzed should be in the "genomes/" folder. The _C. elegans_ (CEL) dataset with its corresponding training and testing partitions is provided in "celdata.tar.gz". 

### OC-SVM

Install the required packages with:
```bash
python -m pip install --user -r src/OC-SVM/requirements.txt
```
The model is trained and tested in a cross-validation scheme with:
```bash
python train_OC-SVM.py
```
Scores are saved in "results/OC-SVM/"

### deeSOM

In a similar way, install the requirements with:
```bash
python -m pip install --user -r src/deeSOM/requirements.txt
```
Then train and test the model with
```bash
python train_deeSOM.py
```

### miRNAss
A python wrapper is used to train and test the model. Install the requirements with:
```bash
python -m pip install --user -r src/miRNAss/requirements.txt
```
If R is already installed, just run:
```bash
python train_miRNAss.py
```
The script also will install the miRNAss package.  

### deepBN
This model was built in a Matlab/Octave framework. Open an Octave terminal in this root folder and run 
```octave
octave:1> train_deepBN
```

### bb-DeepMir
Install the requirements with:
```bash
python -m pip install -r src/bbDeepMir/requirements.txt
```
Train and test the model with:
```bash
KERAS_BACKEND=tensorflow python train_bb-DeepMir.py
```

### bb-deepMiRGene

This method uses libraries that are not available in PiPy. The fastest way to install dependencies is using an Anaconda distribution:
```bash
conda install -c bioconda --file src/bbdeepMiRGene/requirements.txt
```
Train and test the model with:
```bash
KERAS_BACKEND=theano python train_bb-deepMiRGene.py
```
