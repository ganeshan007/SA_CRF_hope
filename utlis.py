import os
import json
import numpy as np
import pandas as pd
from scipy.special import softmax
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from dataset.loader import tag_name_dict
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import normalize, minmax_scale

import matplotlib
if os.environ.get('DISPLAY', '') == '':
    # https://matplotlib.org/faq/usage_faq.html#what-is-a-backend
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def train_and_save_word2vec(corpus_name, tokenized_sentences, wv_dim, wv_epochs):
    model = Word2Vec(size=wv_dim, min_count=1)
    model.build_vocab(tokenized_sentences)
    model.train(tokenized_sentences, total_examples=len(tokenized_sentences), epochs=wv_epochs)
    model.wv.save_word2vec_format('C:/Users/Ganeshan/Desktop/Thesis/CRF_baseline/resource/wv-'+corpus_name+'.bin', binary=True)

    vocabulary = list(model.wv.vocab.keys())

    with open('resource/vocabulary-'+corpus_name+'.txt', 'w') as f:
        for word in vocabulary:
            f.write(word + '\n')

    with open('resource/sentences-'+corpus_name+'.txt', 'w') as f:
        for tokenized_sentence in tokenized_sentences:
            f.write(' '.join(tokenized_sentence) + '\n')

    return vocabulary


def load_word2vec(path, vocabulary, wv_dim, pca_dim):
    model = Word2Vec(size=wv_dim, min_count=1)
    model.build_vocab_from_freq(dict.fromkeys(vocabulary, 1))

    model.intersect_word2vec_format(path, binary=True)

    # Words without an entry in the binary file are silently initialized to random values.
    # We can detect those vectors via their norms which approach zero.
    words_zero_norm = [word for word in model.wv.vocab if np.linalg.norm(model.wv[word]) < 0.05]
    print(' - words not trained: %d / %d' % (len(words_zero_norm), len(vocabulary) - 1), words_zero_norm)

    embeddings = np.zeros((len(vocabulary), wv_dim))
    for index, word in enumerate(vocabulary):
        embeddings[index] = model.wv[word]

    if pca_dim < wv_dim:
        embeddings = PCA(n_components=pca_dim).fit_transform(embeddings)
    embeddings[0] = 0

    return embeddings


def flatten(list_of_list):
    return [item for sublist in list_of_list for item in sublist]


def save_and_plot_history(history, path_to_results):
    with open(path_to_results+'history.json', 'w') as f:
        f.write(json.dumps(history))

    epochs = np.arange(1, len(history['loss'])+1)
    for key in history.keys():
        if key.startswith('val_'):
            k = key.replace('val_', '')

            plt.plot(epochs, history[key])
            plt.title(k)
            plt.ylabel(k)
            plt.xlabel('epoch')

            if k in history.keys():
                plt.plot(epochs, history[k])
                plt.legend(['validation', 'train'])
            else:
                plt.legend(['validation'])

            plt.savefig(path_to_results+k+'.png', dpi=300, bbox_inches='tight')
            plt.clf()


def load_history(path_to_results):
    return json.load(open(path_to_results + 'history.json'))


def save_trans_to_csv(weights, header, corpus_name, path_to_results):
    for idx, trans in enumerate(weights):
        if idx == 2:
            break

        df = pd.DataFrame(
            trans,
            index=header, columns=header
        )
        df.to_csv(path_to_results + 'trans' + str(idx) + '.csv')

        df = pd.DataFrame(
            minmax_scale(trans, axis=1),
            index=header, columns=header
        )
        df.to_csv(path_to_results + 'trans' + str(idx) + '_scale_tag.csv')
        plt.figure(figsize=(13, 10))
        sns.heatmap(df, xticklabels=1, yticklabels=1, center=1)
        plt.savefig(path_to_results + 'trans' + str(idx) + '_scale_tag.png', dpi=300, bbox_inches='tight')
        plt.clf()

        header_name = [tag_name_dict[corpus_name][t] for t in header]
        df = pd.DataFrame(
            minmax_scale(trans, axis=1),
            index=header_name, columns=header_name
        )
        df.to_csv(path_to_results + 'trans' + str(idx) + '_scale_name.csv')
        plt.figure(figsize=(13, 10))
        sns.heatmap(df, xticklabels=1, yticklabels=1, center=1)
        plt.savefig(path_to_results + 'trans' + str(idx) + '_scale_name.png', dpi=300, bbox_inches='tight')
        plt.clf()

        df = pd.DataFrame(
            normalize(trans, norm='l1', axis=1),
            index=header, columns=header
        )
        df.to_csv(path_to_results + 'trans' + str(idx) + '_standard.csv')

        df = pd.DataFrame(
            softmax(trans, axis=1),
            index=header, columns=header
        )
        df.to_csv(path_to_results + 'trans' + str(idx) + '_softmax.csv')

        df = pd.DataFrame(
            np.argsort(np.argsort(-trans, axis=1), axis=1),
            index=header, columns=header
        )
        df.to_csv(path_to_results + 'trans' + str(idx) + '_sort.csv')


class MyLabelBinarizer(LabelBinarizer):
    """
    https://stackoverflow.com/questions/31947140/sklearn-labelbinarizer-returns-vector-when-there-are-2-classes
    """
    def transform(self, y):
        Y = super().transform(y)
        if self.y_type_ == 'binary':
            return np.hstack((Y, 1-Y))
        else:
            return Y

    def inverse_transform(self, Y, threshold=None):
        if self.y_type_ == 'binary':
            return super().inverse_transform(Y[:, 0], threshold)
        else:
            return super().inverse_transform(Y, threshold)


def encode_as_ids(sentence, word2idx):
    output = []
    for word in sentence.split():
        try:
            output.append(word2idx[word])
        except:
            output.append(1)

    return output


def viterbi_vanilla_crf(nodes, trans):
    paths = {(k,): v for k, v in nodes[0].items()}
    for l in range(1, len(nodes)):
        paths_old,paths = paths,{}
        for n,ns in nodes[l].items():
            max_path,max_score = '',-1e10
            for p,ps in paths_old.items():
                score = ns + ps + trans[(p[-1], n)]
                if score > max_score:
                    max_path,max_score = p+(n,), score
            paths[max_path] = max_score
    return max(paths, key=paths.get)


def viterbi_our_crf(nodes, trans0, trans1, spk_change_sequence):
    paths = {(k,): v for k, v in nodes[0].items()}
    for l in range(1, len(nodes)):
        paths_old,paths = paths,{}
        trans = trans0 if spk_change_sequence[l-1] == 0 else trans1
        for n,ns in nodes[l].items():
            max_path,max_score = '',-1e10
            for p,ps in paths_old.items():
                score = ns + ps + trans[(p[-1], n)]
                if score > max_score:
                    max_path,max_score = p+(n,), score
            paths[max_path] = max_score
    return max(paths, key=paths.get)