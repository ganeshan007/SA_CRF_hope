import numpy as np
from math import ceil
import os
import utlis
from utlis import MyLabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ParameterGrid
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
from dataset.loader import get_splits, load_corpus
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


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


corpus_name = 'a2g'

train_set_idx, valid_set_idx, test_set_idx = get_splits(corpus_name)
conversation_list = train_set_idx + valid_set_idx + test_set_idx

corpus, tag_set, speaker_set = load_corpus(corpus_name, conversation_list)

if os.path.isfile('resource/vocabulary-'+corpus_name+'.txt'):
    vocabulary = open('resource/vocabulary-'+corpus_name+'.txt', 'r').read().splitlines()
else:
    train_sentences = [
        sentence.split() for conversation_id in train_set_idx
        for sentence in corpus[conversation_id]['sentence']
    ]
    vocabulary = utlis.train_and_save_word2vec(
        corpus_name,
        train_sentences,
        wv_dim=300,
        wv_epochs=30
    )
vocabulary = ['[PAD]', '[UNK]'] + vocabulary
word_embedding_matrix = utlis.load_word2vec('resource/wv-'+corpus_name+'.bin', vocabulary, wv_dim=300, pca_dim=300)

##########

word2idx = {vocabulary[i]: i for i in range(len(vocabulary))}
for conversation_id in conversation_list:
    corpus[conversation_id]['sequence'] = [
        utlis.encode_as_ids(sentence, word2idx)
        for sentence in corpus[conversation_id]['sentence']
    ]

seq_lens = [len(seq) for cid in conversation_list for seq in corpus[cid]['sequence']]
tag_lb = MyLabelBinarizer().fit(list(tag_set))
spk_le = LabelEncoder().fit(list(speaker_set))
spk_lb = MyLabelBinarizer().fit(range(len(speaker_set)))

for cid in conversation_list:
    corpus[cid]['sequence'] = pad_sequences(corpus[cid]['sequence'], maxlen=max(seq_lens), padding='post', truncating='post')
    corpus[cid]['tag'] = tag_lb.transform(corpus[cid]['tag'])
    corpus[cid]['speaker'] = spk_le.transform(corpus[cid]['speaker'])
    corpus[cid]['speaker_change'] = np.not_equal(corpus[cid]['speaker'][:-1], corpus[cid]['speaker'][1:]).astype(int)
    corpus[cid]['speaker'] = spk_lb.transform(corpus[cid]['speaker'])
key='test'
X, Y, SPK, SPK_C = dict(), dict(), dict(), dict()
for value in test_set_idx:
    X[key] = [corpus[cid]['sequence'] for cid in value]
    Y[key] = [corpus[cid]['tag'] for cid in value]
    SPK[key] = [corpus[cid]['speaker'] for cid in value]
    SPK_C[key] = [corpus[cid]['speaker_change'] for cid in value]
mode='our_crf-spk_c'
batch_size=4
encoder_type='lstm'
n_hidden=300
dropout_rate=0.2
n_tags = len(tag_lb.classes_)
n_spks = len(spk_lb.classes_)
crf_lr_multiplier=1
test_data = data_generator('test', X, Y, SPK, SPK_C, mode, batch_size)
input_X = Input(shape=(None, None), dtype='int32')
s2v_module = get_s2v_module(encoder_type, word_embedding_matrix, n_hidden, dropout_rate)
output = TimeDistributed(s2v_module)(input_X)

bilstm_layer = Bidirectional(
    LSTM(units=n_hidden,
            activation='tanh',
            return_sequences=True)
)
dropout_layer = Dropout(dropout_rate)



input_SPK_C = Input(shape=(None,), dtype='int32')
dense_layer_crf = Dense(units=n_tags if batch_size == 1 else n_tags + 1)
crf = OurCRF(ignore_last_label=False if batch_size == 1 else True)
output = crf(dense_layer_crf(dropout_layer(bilstm_layer(output))))

model = Model([input_X, input_SPK_C], output)
model.compile(optimizer=LRMultiplier('adam', {'our_crf': crf_lr_multiplier}), loss=crf.loss_wrapper(input_SPK_C), metrics=[])



# import keras

model.load_weights('5.h5')
print(model.summary())
# predictions=model.predict_generator(test_data,steps=10)
# print(predictions)
# print(X)
model.predict(X,steps=1)