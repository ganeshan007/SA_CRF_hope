import os
import csv
import json
import utlis
import string
import pickle
import secrets
import trainer
import numpy as np
from math import ceil
from nltk.util import ngrams
from datetime import datetime
from keras import backend as K
from utlis import MyLabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ParameterGrid
from dataset.loader import get_splits, load_corpus
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

corpus_name = 'hope'

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
    if cid!='107':
        corpus[cid]['sequence'] = pad_sequences(corpus[cid]['sequence'], maxlen=max(seq_lens), padding='post', truncating='post')
        corpus[cid]['tag'] = tag_lb.transform(corpus[cid]['tag'])
        corpus[cid]['speaker'] = spk_le.transform(corpus[cid]['speaker'])
        corpus[cid]['speaker_change'] = np.not_equal(corpus[cid]['speaker'][:-1], corpus[cid]['speaker'][1:]).astype(int)
        corpus[cid]['speaker'] = spk_lb.transform(corpus[cid]['speaker'])

X, Y, SPK, SPK_C = dict(), dict(), dict(), dict()
for key, value in zip(['train', 'valid', 'test'], [train_set_idx, valid_set_idx, test_set_idx]):
    X[key] = [corpus[cid]['sequence'] for cid in value]
    Y[key] = [corpus[cid]['tag'] for cid in value]
    SPK[key] = [corpus[cid]['speaker'] for cid in value]
    SPK_C[key] = [corpus[cid]['speaker_change'] for cid in value]

##########

param_grid = {
    'encoder_type': ['lstm'],  # lstm, bilstm, att-bilstm
    'mode': ['our_crf-spk_c'],
    'batch_size': [4],
    'dropout_rate': [0.2],
    'crf_lr_multiplier': [1]
}

for param in ParameterGrid(param_grid):
    encoder_type = param['encoder_type']
    mode = param['mode']
    batch_size = param['batch_size']
    dropout_rate = param['dropout_rate']
    crf_lr_multiplier = param['crf_lr_multiplier']

    f_id = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(5))
    path_to_results = 'Results/' 
    os.makedirs(path_to_results + 'model_on_epoch_end')
    print('path_to_results', path_to_results)

    n_tags = len(tag_lb.classes_)
    n_spks = len(spk_lb.classes_)
    history, model = trainer.train(X, Y, SPK, SPK_C, encoder_type, word_embedding_matrix, tag_lb, n_tags, n_spks, batch_size, dropout_rate, crf_lr_multiplier, mode, path_to_results)
    utlis.save_and_plot_history(history, path_to_results)

    ##########

    val_viterbi_accuracy_list = np.array(history['val_viterbi_accuracy'])
    best_epoch = int(np.where(val_viterbi_accuracy_list == val_viterbi_accuracy_list.max())[0][-1] + 1)
    val_loss = np.array(history['val_loss'])[best_epoch-1]

    print('the best epoch based on val_viterbi_accuracy:', best_epoch)
    model.load_weights(path_to_results + 'model_on_epoch_end/' + str(best_epoch) + '.h5')
    n_test_samples = len(X['test'])

    if mode == 'vanilla_crf' or mode == 'vanilla_crf-spk' or mode == 'vanilla_crf-spk_c':
        trans = {}
        for i in range(n_tags):
            for j in range(n_tags):
                tag_from = tag_lb.classes_[i]
                tag_to = tag_lb.classes_[j]
                trans[(tag_from, tag_to)] = model.get_layer('vanilla_crf_1').get_weights()[0][i, j]
        utlis.save_trans_to_csv(model.get_layer('vanilla_crf_1').get_weights(), tag_lb.classes_, corpus_name, path_to_results)

        for i in range(n_test_samples):
            if mode == 'vanilla_crf':
                probas = model.predict(np.array([X['test'][i]]))[0]
            if mode == 'vanilla_crf-spk':
                probas = model.predict([np.array([X['test'][i]]), np.array([SPK['test'][i]])])[0]
            if mode == 'vanilla_crf-spk_c':
                probas = model.predict([np.array([X['test'][i]]), np.expand_dims(np.array([
                    np.concatenate([[1], SPK_C['test'][i]])]), axis=-1)])[0]
            nodes = [dict(zip(tag_lb.classes_, j)) for j in probas[:, :n_tags]]
            tags = utlis.viterbi_vanilla_crf(nodes, trans)

            corpus[test_set_idx[i]]['prediction'] = list(tags)
            corpus[test_set_idx[i]]['tag'] = list(tag_lb.inverse_transform(Y['test'][i]))

    if mode == 'our_crf-spk_c':
        trans0, trans1 = {}, {}
        for i in range(n_tags):
            for j in range(n_tags):
                tag_from = tag_lb.classes_[i]
                tag_to = tag_lb.classes_[j]
                trans0[(tag_from, tag_to)] = model.get_layer('our_crf_1').get_weights()[0][i, j]
                trans1[(tag_from, tag_to)] = model.get_layer('our_crf_1').get_weights()[1][i, j]
        utlis.save_trans_to_csv(model.get_layer('our_crf_1').get_weights(), tag_lb.classes_, corpus_name, path_to_results)

        for i in range(n_test_samples):
            probas = model.predict([np.array([X['test'][i]]), np.array([SPK_C['test'][i]])])[0]
            nodes = [dict(zip(tag_lb.classes_, j)) for j in probas[:, :n_tags]]
            tags = utlis.viterbi_our_crf(nodes, trans0, trans1, SPK_C['test'][i])

            corpus[test_set_idx[i]]['prediction'] = list(tags)
            corpus[test_set_idx[i]]['tag'] = list(tag_lb.inverse_transform(Y['test'][i]))

    ##########

    y_true = []
    y_pred = []
    for conversation_id in test_set_idx:
        y_true = y_true + corpus[conversation_id]['tag']
        y_pred = y_pred + corpus[conversation_id]['prediction']
    wf1 = f1_score(y_true,y_pred,average='weighted')
    macrof1 = f1_score(y_true,y_pred,average='macro')
    microf1 = f1_score(y_true,y_pred,average='micro')
    print('weighted, macro, micro {}, {}, {}'.format(wf1,macrof1,microf1))

    unigram_accuracy = accuracy_score(y_pred=y_pred, y_true=y_true)
    labels = list(set(y_true)|set(y_pred))
    matrix = confusion_matrix(y_pred=y_pred, y_true=y_true, labels=labels)
    unigram_accuracy_per_label = sorted(zip(labels, np.nan_to_num(matrix.diagonal() / matrix.sum(axis=1)), matrix.diagonal(), matrix.sum(axis=1)),
                                        key=lambda x: x[1], reverse=True)

    y_true = []
    y_pred = []
    for conversation_id in test_set_idx:
        y_true = y_true + ['->'.join(gram) for gram in ngrams(corpus[conversation_id]['tag'], 2)]
        y_pred = y_pred + ['->'.join(gram) for gram in ngrams(corpus[conversation_id]['prediction'], 2)]
    bigram_accuracy = accuracy_score(y_pred=y_pred, y_true=y_true)
    labels = list(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_pred=y_pred, y_true=y_true, labels=labels)
    bigram_accuracy_per_label = sorted(zip(labels, np.nan_to_num(matrix.diagonal() / matrix.sum(axis=1)), matrix.diagonal(), matrix.sum(axis=1)),
                                       key=lambda x: x[1], reverse=True)

    y_true = []
    y_pred = []
    for conversation_id in test_set_idx:
        y_true = y_true + ['->'.join(gram) for gram in ngrams(corpus[conversation_id]['tag'], 3)]
        y_pred = y_pred + ['->'.join(gram) for gram in ngrams(corpus[conversation_id]['prediction'], 3)]
    trigram_accuracy = accuracy_score(y_pred=y_pred, y_true=y_true)
    labels = list(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_pred=y_pred, y_true=y_true, labels=labels)
    trigram_accuracy_per_label = sorted(zip(labels, np.nan_to_num(matrix.diagonal() / matrix.sum(axis=1)), matrix.diagonal(), matrix.sum(axis=1)),
                                        key=lambda x: x[1], reverse=True)

    print(unigram_accuracy, bigram_accuracy, trigram_accuracy)

    pickle.dump({cid: corpus[cid] for cid in test_set_idx}, open(path_to_results + 'prediction.pkl', 'wb'))
    with open(path_to_results+'unigram_accuracy_per_label.csv', 'w') as out:
        csv_out = csv.writer(out)
        for row in unigram_accuracy_per_label:
            csv_out.writerow(row)
    with open(path_to_results+'bigram_accuracy_per_label.csv', 'w') as out:
        csv_out = csv.writer(out)
        for row in bigram_accuracy_per_label:
            csv_out.writerow(row)
    with open(path_to_results+'trigram_accuracy_per_label.csv', 'w') as out:
        csv_out = csv.writer(out)
        for row in trigram_accuracy_per_label:
            csv_out.writerow(row)

    with open(path_to_results + 'result.json', 'w') as f:
        f.write(json.dumps(
            dict(((k, eval(k)) for k in ('corpus_name', 'encoder_type', 'mode', 'batch_size', 'dropout_rate', 'crf_lr_multiplier', 'best_epoch', 'val_loss', 'unigram_accuracy', 'bigram_accuracy', 'trigram_accuracy')))
        ))

    K.clear_session()
