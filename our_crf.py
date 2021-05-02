# -*- coding:utf-8 -*-
import utlis
import numpy as np
import keras.backend as K
from keras.layers import Layer
from keras.callbacks import Callback
from sklearn.metrics import accuracy_score

class OurCRF(Layer):
    def __init__(self, ignore_last_label=False, **kwargs):
        """ignore_last_label：定义要不要忽略最后一个标签，起到mask的效果
        """
        self.ignore_last_label = 1 if ignore_last_label else 0
        super(OurCRF, self).__init__(**kwargs)

    def build(self, input_shape):
        self.num_labels = input_shape[-1] - self.ignore_last_label
        self.trans0 = self.add_weight(name='crf_trans0',
                                     shape=(self.num_labels, self.num_labels),
                                     initializer='glorot_uniform',
                                     trainable=True)
        self.trans1 = self.add_weight(name='crf_trans1',
                                     shape=(self.num_labels, self.num_labels),
                                     initializer='glorot_uniform',
                                     trainable=True)
        self.trans2 = self.add_weight(name='crf_trans2',
                                     shape=(self.num_labels, self.num_labels),
                                     initializer='zeros',
                                     trainable=False)

    def log_norm_step(self, inputs, states):
        """递归计算归一化因子
        要点：1、递归计算；2、用logsumexp避免溢出。
        技巧：通过expand_dims来对齐张量。
        """
        spk_change_sequence = states[1]
        spk_change = spk_change_sequence[:, 0]

        states = K.expand_dims(states[0], 2) # (batch_size, output_dim, 1)
        # trans = K.expand_dims(self.trans, 0) # (1, output_dim, output_dim)
        trans = K.gather(K.stack([self.trans0, self.trans1, self.trans2]), spk_change)
        output = K.logsumexp(states+trans, 1) # (batch_size, output_dim)
        return output+inputs, [output+inputs, K.concatenate([spk_change_sequence[:, 1:], spk_change_sequence[:, 0:1]])]

    def path_score(self, inputs, labels, spk_change_sequence):
        """计算目标路径的相对概率（还没有归一化）
        要点：逐标签得分，加上转移概率得分。
        技巧：用“预测”点乘“目标”的方法抽取出目标路径的得分。
        """
        point_score = K.sum(K.sum(inputs*labels, 2), 1, keepdims=True) # 逐标签得分
        labels1 = K.expand_dims(labels[:, :-1], 3)
        labels2 = K.expand_dims(labels[:, 1:], 2)
        labels = labels1 * labels2 # 两个错位labels，负责从转移矩阵中抽取目标转移得分
        # trans = K.expand_dims(K.expand_dims(self.trans, 0), 0)
        trans = K.gather(K.stack([self.trans0, self.trans1, self.trans2]), spk_change_sequence)
        trans_score = K.sum(K.sum(trans*labels, [2,3]), 1, keepdims=True)
        return point_score+trans_score # 两部分得分之和

    def call(self, inputs): # CRF本身不改变输出，它只是一个loss
        return inputs

    def loss_wrapper(self, spk_change_sequence):
        def loss(y_true, y_pred): # 目标y_pred需要是one hot形式
            mask = 1-y_true[:,1:,-1] if self.ignore_last_label else None
            y_true,y_pred = y_true[:,:,:self.num_labels],y_pred[:,:,:self.num_labels]
            init_states = [y_pred[:,0], spk_change_sequence] # 初始状态
            log_norm,_,_ = K.rnn(self.log_norm_step, y_pred[:,1:], init_states, mask=mask) # 计算Z向量（对数）
            log_norm = K.logsumexp(log_norm, 1, keepdims=True) # 计算Z（对数）
            path_score = self.path_score(y_pred, y_true, spk_change_sequence) # 计算分子（对数）
            return log_norm - path_score # 即log(分子/分母)
        return loss

    def accuracy(self, y_true, y_pred): # 训练过程中显示逐帧准确率的函数，排除了mask的影响
        mask = 1-y_true[:,:,-1] if self.ignore_last_label else None
        y_true,y_pred = y_true[:,:,:self.num_labels],y_pred[:,:,:self.num_labels]
        isequal = K.equal(K.argmax(y_true, 2), K.argmax(y_pred, 2))
        isequal = K.cast(isequal, 'float32')
        if mask == None:
            return K.mean(isequal)
        else:
            return K.sum(isequal*mask) / K.sum(mask)


class ViterbiAccuracy_OurCRF(Callback):
    def __init__(self, validation_data, validation_steps, tag_lb, n_tags):
        super().__init__()
        self.validation_data = validation_data
        self.validation_steps = validation_steps
        self.tag_lb = tag_lb
        self.n_tags = n_tags

    def on_epoch_end(self, epoch, logs={}):
        trans0, trans1 = {}, {}
        for i in range(self.n_tags):
            for j in range(self.n_tags):
                tag_from = self.tag_lb.classes_[i]
                tag_to = self.tag_lb.classes_[j]
                trans0[(tag_from, tag_to)] = self.model.get_layer('our_crf_1').get_weights()[0][i, j]
                trans1[(tag_from, tag_to)] = self.model.get_layer('our_crf_1').get_weights()[1][i, j]

        y_pred, y_true = [], []
        for _ in range(self.validation_steps):
            [B_X, B_SPK_C], B_Y = next(self.validation_data)
            for i in range(len(B_X)):
                probas = self.model.predict([np.array([B_X[i]]), np.array([B_SPK_C[i]])])[0]
                nodes = [dict(zip(self.tag_lb.classes_, j)) for j in probas[:, :self.n_tags]]
                tags = utlis.viterbi_our_crf(nodes, trans0, trans1, B_SPK_C[i])

                y_pred = y_pred + list(tags)
                y_true = y_true + list(self.tag_lb.inverse_transform(B_Y[i]))

        accuracy = accuracy_score(y_pred=y_pred, y_true=y_true)
        print('val_viterbi_accuracy', accuracy)

        logs['val_viterbi_accuracy'] = accuracy