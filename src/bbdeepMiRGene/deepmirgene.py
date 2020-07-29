# ============================================================
# Leandro Bugnon, lbugnon@sinc.unl.edu.ar
# sinc(i) - http://www.sinc.unl.edu.ar
# ============================================================
# code from https://github.com/eleventh83/deepMiRGene

from Bio import SeqIO  ## fasta read
import RNA ## RNAFold
import re ## reg to find loops
from sklearn.model_selection import KFold  ## for cv
from sklearn import metrics ## evaluation
import pickle
import numpy as np
# keras
from keras.models import Model
from keras.layers import Input, LSTM, TimeDistributed, Dropout, Dense, Permute, Flatten, Multiply, RepeatVector, Activation, Masking
from keras import regularizers, optimizers
from keras.preprocessing.sequence import pad_sequences
from keras.layers.wrappers import Wrapper
from keras.engine.topology import InputSpec
from keras import backend as K



def one_hot(X_enc,DIM_ENC):
    X_enc_len = len(X_enc)
    X_enc_vec = np.zeros((X_enc_len, DIM_ENC))
    X_enc_vec[np.arange(np.nonzero(X_enc)[0][0],X_enc_len), np.int32([X_enc[k]-1 for k in np.nonzero(X_enc)[0].tolist()])] = 1
    return X_enc_vec.tolist()


def one_hot_wrap(X_encs, MAX_LEN, DIM_ENC):
    num_X_encs = len(X_encs)
    X_encs_padded = pad_sequences(X_encs, maxlen = MAX_LEN, dtype='int8')
    X_encs_ = np.zeros(num_X_encs).tolist()
    for i in range(num_X_encs):
        if i%10000==0:
            print(i)
        X_encs_[i] = one_hot(X_encs_padded[i],DIM_ENC)
    return np.int32(X_encs_)


def str2num(a_str):
    ints_ = [0]*len(a_str)
    for i, c in enumerate(a_str.lower()):
        if c == ')':
            ints_[i] = 1
        elif c == '.':
            ints_[i] = 2
        elif c == ':':
            ints_[i] = 3
    return ints_


def seq2num(a_seq):
    ints_ = [0]*len(a_seq)
    for i, c in enumerate(a_seq.lower()):
        if c == 'c':
            ints_[i] = 1
        elif c == 'g':
            ints_[i] = 2
        elif (c == 'u') | (c == 't'):
            ints_[i] = 3
    return ints_

# convert loops from '.'s to ':'s
def convloops(a_str):
    chrs_ = a_str[:]
    prog = re.compile('\(+(\.+)\)+')
    for m in prog.finditer(a_str):
        #print m.start(), m.group()
        chrs_ = "".join((chrs_[:m.regs[1][0]],':'*(m.regs[1][1]-m.regs[1][0]),chrs_[(m.regs[1][1]):]))
    return chrs_


def encode(seqs,strs):
    if not isinstance(seqs, list):
        print("[ERROR:encode] Input type must be multidimensional list.")
        return
 
    # if len(seqs) != len(strs) :
    #     print("[ERROR:encode] # sequences must be equal to # structures.")
    #     return
 
    encs = []
    if strs is not None:
        for a_seq, a_str in zip(seqs,strs):
            encs.append([4*i_seq+i_str+1 for i_seq, i_str in zip(seq2num(a_seq), str2num(convloops(a_str)))])
    else:
        for a_seq in seqs:
            encs.append([4*i_seq+1 for i_seq in seq2num(a_seq)])
    return encs

def seq2str(seqs):
    return [RNA.fold(a_seq)[0] for a_seq in seqs]

def make_safe(x):
    return K.clip(x, K.common._EPSILON, 1.0 - K.common._EPSILON)

def import_data(seqs, use_secondary_structure=True):
    if type(seqs) != list:
        seqs = seqs.tolist()
    if use_secondary_structure:
        strs = seq2str(seqs)  # convert to (). format
    else:
        strs = None
    return encode(seqs, strs)

class ProbabilityTensor(Wrapper):
    def __init__(self, dense_function=None, *args, **kwargs):
        self.supports_masking = True
        self.input_spec = [InputSpec(ndim=3)]
        layer = TimeDistributed(Dense(1, name='ptensor_func'))
        super(ProbabilityTensor, self).__init__(layer, *args, **kwargs)
 
    def build(self, input_shape):
        assert len(input_shape) == 3
        self.input_spec = [InputSpec(shape=input_shape)]
        #if K._BACKEND == 'tensorflow':
        if not input_shape[1]:
            raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis.')
   
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
  
        super(ProbabilityTensor, self).build()
 
    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, (list,tuple)) and not isinstance(input_shape[0], int):
            input_shape = input_shape[0]
        return (input_shape[0], input_shape[1])
 
    def squash_mask(self, mask):
        if K.ndim(mask) == 2:
            return mask
        elif K.ndim(mask) == 3:
            return K.any(mask, axis=-1)
 
    def compute_mask(self, x, mask=None):
        if mask is None:
            return None
        return self.squash_mask(mask)
 
    def call(self, x, mask=None):
        energy = K.squeeze(self.layer(x), 2)
        p_matrix = K.softmax(energy) ## (nb_sample, time)
        if mask is not None:
            mask = self.squash_mask(mask)
   
        p_matrix = make_safe(p_matrix * mask)
        p_matrix = (p_matrix / K.sum(p_matrix, axis=-1, keepdims=True))*mask
        return p_matrix
 
    def get_config(self):
        config = {}
        base_config = super(ProbabilityTensor, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))





class SoftAttention(ProbabilityTensor):
    def compute_output_shape(self, input_shape):
        return [(input_shape[0], input_shape[1] * input_shape[2]), (input_shape[0], input_shape[1])]
 
    def compute_mask(self, x, mask=None):
        if mask is None or mask.ndim==2:
            return [None, None]
        else:
            raise Exception("Unexpected situation")
 
    def call(self, x, mask=None):
        p_vector = super(SoftAttention, self).call(x, mask)
        p_vectors = K.expand_dims(p_vector, 2)
        expanded_p = K.repeat_elements(p_vectors, K.shape(x)[2], axis=2)
        mul = expanded_p * x
  
        return [K.reshape(mul,[K.shape(x)[0],K.shape(x)[1]*K.shape(x)[2]]), p_vector]


def mymodel():

    # parameters
    MAX_LEN = 400 # maximum sequence length
    DIM_ENC = 16 # dimension of a one-hot encoded vector (e.g., 4 (sequence) x 4 (structure) = 16)
    DIM_LSTM1 = 20
    DIM_LSTM2 = 10
    DIM_DENSE1 = 400
    DMI_DENSE2 = 100
    
    inputs = Input(shape=(MAX_LEN, DIM_ENC), name='inputs')
    msk = Masking(mask_value=0)(inputs)
    lstm1 = LSTM(DIM_LSTM1, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)(msk)
    lstm2 = LSTM(DIM_LSTM2, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)(lstm1)
 
    att, pv = SoftAttention(lstm2)(lstm2)
 
    do1 = Dropout(0.1)(att)
    dense1 = Dense(DIM_DENSE1,activation='sigmoid')(do1)
    do2 = Dropout(0.1)(dense1)
    dense2 = Dense(DMI_DENSE2,activation='sigmoid')(do2)
    outputs = Dense(2,activation='softmax')(dense2)
 
    model=Model(outputs=outputs, inputs=inputs)
    model.compile(optimizer='adam', loss='binary_crossentropy')
 
    return model
