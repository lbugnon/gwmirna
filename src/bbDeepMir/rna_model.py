# ============================================================
# Leandro Bugnon, lbugnon@sinc.unl.edu.ar
# sinc(i) - http://www.sinc.unl.edu.ar
# ============================================================
# Adapted from https://github.com/HubertTang/DeepMir

import keras
from keras.utils import to_categorical
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, concatenate, BatchNormalization, \
    Activation, AveragePooling2D
from keras import optimizers

# ================= model using to train one-hot dataset ==================
def DeepRfam(seq_length=312, num_c=4, num_filters=256,
             filter_sizes=[24, 36, 48, 60, 72, 84, 96, 108],
             dropout_rate=0.5, num_classes=143, num_hidden=512):
    # initialization
    in_shape = (seq_length, num_c, 1)

    input_shape = Input(shape=in_shape)
   
    pooled_outputs = []
    for i in range(len(filter_sizes)):
        conv = Conv2D(num_filters, (filter_sizes[i], num_c), padding='valid', activation='relu')(input_shape)
        pool = MaxPooling2D((seq_length - filter_sizes[i] + 1, 1),
                            padding='valid')(conv)
        pooled_outputs.append(pool)

    merge = concatenate(pooled_outputs)

    x = Flatten()(merge)
    x = Dropout(dropout_rate)(x)
    x = Dense(num_hidden, activation='relu')(x)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(input_shape, out)
    return model


def train(train_dataset,  out_dir, fold, seq_length, dataset_name, n_iter):
    model = DeepRfam(
        seq_length=seq_length,
        num_classes=2)
    print(model.summary())

    # model compile
    model.compile(
        loss="categorical_crossentropy",
        optimizer=eval(f"optimizers.Adam")(lr=0.001),
        metrics=['accuracy']
    )

    # start and record training history
    print("Train start fold %d" % fold)
    train_history = model.fit_generator(train_dataset, epochs=n_iter, verbose=1,
                                        use_multiprocessing=True, workers=8)
    print("Train end")
    return model



