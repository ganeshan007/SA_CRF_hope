import numpy as np
from math import ceil
from sklearn.utils import shuffle
from keras.models import Model
from keras.initializers import Constant
from keras.constraints import UnitNorm
from keras.layers import Input, Dense, Embedding, LSTM, Dropout
from keras.layers import TimeDistributed, Bidirectional, concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping
from vanilla_crf import VanillaCRF, ViterbiAccuracy_VanillaCRF
from our_crf import OurCRF, ViterbiAccuracy_OurCRF
from attention_with_context import AttentionWithContext
from keras_lr_multiplier import LRMultiplier

def data_generator(set_name, X, Y, SPK, SPK_C, mode, batch_size):
    n_samples = len(X[set_name])
    while True:
        X[set_name], Y[set_name], SPK[set_name], SPK_C[set_name] = shuffle(X[set_name], Y[set_name], SPK[set_name], SPK_C[set_name])

        B_X, B_Y, B_SPK, B_SPK_C = [], [], [], []

        for i in range(n_samples):
            B_X.append(X[set_name][i])
            B_Y.append(Y[set_name][i])
            B_SPK.append(SPK[set_name][i])
            B_SPK_C.append(SPK_C[set_name][i])

            if len(B_X) == batch_size or i == n_samples - 1:
                if len(B_X) > 1:

                    max_len = max([len(x) for x in B_X])
                    for j in range(len(B_X)):
                        current_len = len(B_X[j])
                        if current_len < max_len:
                            pad = np.zeros((max_len-current_len, B_X[j].shape[1]))
                            pad[:, 0] = 1
                            B_X[j] = np.vstack([B_X[j], pad])

                    max_len = max([len(y) for y in B_Y])
                    for j in range(len(B_Y)):
                        current_len = len(B_Y[j])
                        pad = np.zeros((current_len, 1))
                        B_Y[j] = np.concatenate([B_Y[j], pad], axis=1)
                        if current_len < max_len:
                            pad = np.zeros((max_len-current_len, B_Y[j].shape[1]))
                            pad[:, -1] = 1
                            B_Y[j] = np.vstack([B_Y[j], pad])

                    max_len = max([len(spk) for spk in B_SPK])
                    for j in range(len(B_SPK)):
                        current_len = len(B_SPK[j])
                        if current_len < max_len:
                            pad = np.zeros((max_len-current_len, B_SPK[j].shape[1]))
                            B_SPK[j] = np.vstack([B_SPK[j], pad])

                    max_len = max([len(spk_c) for spk_c in B_SPK_C])
                    for j in range(len(B_SPK_C)):
                        current_len = len(B_SPK_C[j])
                        if current_len < max_len:
                            pad = np.zeros(max_len-current_len)
                            pad = pad + 2
                            B_SPK_C[j] = np.concatenate([B_SPK_C[j], pad])

                if mode == 'vanilla_crf':
                    yield (np.array(B_X),
                           np.array(B_Y))
                if mode == 'vanilla_crf-spk':
                    yield ([np.array(B_X), np.array(B_SPK)],
                           np.array(B_Y))
                if mode == 'vanilla_crf-spk_c':
                    B_SPK_C = np.array(B_SPK_C)
                    pad = np.ones((B_SPK_C.shape[0], 1))
                    B_SPK_C = np.concatenate([pad, B_SPK_C], axis=-1)
                    yield ([np.array(B_X), np.expand_dims(B_SPK_C, axis=-1)],
                           np.array(B_Y))
                if mode == 'our_crf-spk_c':
                    yield ([np.array(B_X), np.array(B_SPK_C)],
                           np.array(B_Y))

                B_X, B_Y, B_SPK, B_SPK_C = [], [], [], []


def get_s2v_module(encoder_type, word_embedding_matrix, n_hidden, dropout_rate):
    input = Input(shape=(None,), dtype='int32')

    embedding_layer = Embedding(
        word_embedding_matrix.shape[0],
        word_embedding_matrix.shape[1],
        embeddings_initializer=Constant(word_embedding_matrix),
        trainable=False,
        mask_zero=True
    )

    output = embedding_layer(input)

    if encoder_type == 'lstm':
        lstm_layer = LSTM(
            units=n_hidden,
            activation='tanh',
            return_sequences=False
        )
        output = lstm_layer(output)

    if encoder_type == 'bilstm':
        bilstm_layer = Bidirectional(
            LSTM(units=n_hidden,
                 activation='tanh',
                 return_sequences=False)
        )
        output = bilstm_layer(output)

    if encoder_type == 'att-bilstm':
        bilstm_layer = Bidirectional(
            LSTM(units=n_hidden,
                 activation='tanh',
                 return_sequences=True)
        )
        attention_layer = AttentionWithContext(u_constraint=UnitNorm())
        output = attention_layer(bilstm_layer(output))

    dropout_layer = Dropout(dropout_rate)

    output = dropout_layer(output)

    model = Model(input, output)
    model.summary()

    return model


def train(X, Y, SPK, SPK_C, encoder_type, word_embedding_matrix, tag_lb, n_tags, n_spks, batch_size, dropout_rate, crf_lr_multiplier, mode, path_to_results):
    epochs = 20
    n_hidden = 300
    n_train_samples = len(X['train'])
    n_valid_samples = len(X['valid'])
    validation_data = data_generator('valid', X, Y, SPK, SPK_C, mode, batch_size)
    validation_steps = ceil(n_valid_samples / batch_size)


    callbacks = [ModelCheckpoint(filepath=path_to_results+'model_on_epoch_end/'+'{epoch}.h5',
                                 save_weights_only=True),
                 EarlyStopping(monitor='val_viterbi_accuracy', patience=20)]

    input_X = Input(shape=(None, None), dtype='int32')
    s2v_module = get_s2v_module(encoder_type, word_embedding_matrix, n_hidden, dropout_rate)
    output = TimeDistributed(s2v_module)(input_X)

    bilstm_layer = Bidirectional(
        LSTM(units=n_hidden,
             activation='tanh',
             return_sequences=True)
    )
    dropout_layer = Dropout(dropout_rate)

    if mode == 'vanilla_crf':
        dense_layer_crf = Dense(units=n_tags if batch_size == 1 else n_tags+1)
        crf = VanillaCRF(ignore_last_label=False if batch_size == 1 else True)
        output = crf(dense_layer_crf(dropout_layer(bilstm_layer(output))))

        model = Model(input_X, output)
        model.compile(optimizer=LRMultiplier(name='adam', multipliers={'vanilla_crf': crf_lr_multiplier}), loss=crf.loss, metrics=[])
        metric_callback = ViterbiAccuracy_VanillaCRF(validation_data, validation_steps, tag_lb, n_tags, mode)

    if mode == 'vanilla_crf-spk':
        input_SPK = Input(shape=(None, n_spks), dtype='float32')
        dense_layer_crf = Dense(units=n_tags if batch_size == 1 else n_tags+1)
        crf = VanillaCRF(ignore_last_label=False if batch_size == 1 else True)
        output = crf(dense_layer_crf(dropout_layer(bilstm_layer(concatenate([input_SPK, output])))))

        model = Model([input_X, input_SPK], output)
        model.compile(optimizer=LRMultiplier('adam', {'vanilla_crf': crf_lr_multiplier}), loss=crf.loss, metrics=[])
        metric_callback = ViterbiAccuracy_VanillaCRF(validation_data, validation_steps, tag_lb, n_tags, mode)

    if mode == 'vanilla_crf-spk_c':
        input_SPK_C = Input(shape=(None, 1), dtype='float32')
        dense_layer_crf = Dense(units=n_tags if batch_size == 1 else n_tags+1)
        crf = VanillaCRF(ignore_last_label=False if batch_size == 1 else True)
        output = crf(dense_layer_crf(dropout_layer(bilstm_layer(concatenate([input_SPK_C, output])))))

        model = Model([input_X, input_SPK_C], output)
        model.compile(optimizer=LRMultiplier('adam', {'vanilla_crf': crf_lr_multiplier}), loss=crf.loss, metrics=[])
        metric_callback = ViterbiAccuracy_VanillaCRF(validation_data, validation_steps, tag_lb, n_tags, mode)

    if mode == 'our_crf-spk_c':
        input_SPK_C = Input(shape=(None,), dtype='int32')
        dense_layer_crf = Dense(units=n_tags if batch_size == 1 else n_tags + 1)
        crf = OurCRF(ignore_last_label=False if batch_size == 1 else True)
        output = crf(dense_layer_crf(dropout_layer(bilstm_layer(output))))

        model = Model([input_X, input_SPK_C], output)
        model.compile(optimizer=LRMultiplier(optimizer='adam', multipliers={'our_crf': crf_lr_multiplier}), loss=crf.loss_wrapper(input_SPK_C), metrics=[])
        metric_callback = ViterbiAccuracy_OurCRF(validation_data, validation_steps, tag_lb, n_tags)

    model.summary()
    callbacks = [metric_callback] + callbacks
    history = model.fit_generator(
        data_generator('train', X, Y, SPK, SPK_C, mode, batch_size),
        steps_per_epoch=ceil(n_train_samples/batch_size),
        epochs=epochs,
        validation_data=validation_data,
        validation_steps=validation_steps,
        callbacks=callbacks
    )

    return history.history, model


