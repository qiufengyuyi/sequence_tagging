import tensorflow as tf
from tensorflow.contrib import crf
from tensorflow.contrib import rnn
from tensorflow.contrib.rnn.python.ops.rnn import stack_bidirectional_dynamic_rnn

class BLSTM_CRF(object):
    def __init__(self, embedded_chars, hidden_unit,rnn_size, num_layers, dropout_rate,
                 num_labels, labels, lengths, is_training):
        """
        BLSTM-CRF 网络
        :param embedded_chars: Fine-tuning embedding input
        :param hidden_unit: LSTM的隐含单元个数
        :param cell_type: RNN类型（LSTM OR GRU DICNN will be add in feature）
        :param num_layers: RNN的层数
        :param droupout_rate: droupout rate
        :param num_labels: 标签数量
        :param labels: 真实标签
        :param lengths: [batch_size] 每个batch下序列的真实长度
        :param is_training: 是否是训练过程
        """
        self.hidden_unit = hidden_unit
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.embedded_chars = embedded_chars
        self.num_labels = num_labels
        self.labels = labels
        self.lengths = lengths
        self.embedding_dims = embedded_chars.shape[-1].value
        self.is_training = is_training
        self.rnn_size = rnn_size

    def add_blstm_crf_layer(self):
        """
        blstm-crf网络
        :return:
        """
        if self.is_training:
            self.embedded_chars = tf.nn.dropout(self.embedded_chars, self.dropout_rate)

        # blstm
        lstm_output = self.blstm_layer(self.embedded_chars)
        # project
        logits = tf.layers.dense(lstm_output, self.num_labels)
        # crf
        loss, trans = self.crf_layer(logits)
        # CRF decode, pred_ids 是一条最大概率的标注路径
        pred_ids, _ = crf.crf_decode(potentials=logits, transition_params=trans, sequence_length=self.lengths)



        # acc = tf.metrics.accuracy(self.labels, pred_ids, weights)
        # def metric_fn(label_ids, pred_ids):
        #     return precision(label_ids, pred_ids, self.num_labels),recall(label_ids, pred_ids, self.num_labels),f1(label_ids, pred_ids, self.num_labels)
        # precisoin_score,recall_score,f1_score = metric_fn(self.labels,pred_ids)
        return loss,logits,trans,pred_ids
        # return logits

    def get_rnn_cell(self):
        decoder_cell = tf.contrib.rnn.LSTMCell(self.rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=10))
        decoder_cell = tf.contrib.rnn.DropoutWrapper(
            cell=decoder_cell, input_keep_prob=self.dropout_rate)
        # decoder_cell = tf.contrib.rnn.ResidualWrapper(
        # decoder_cell, residual_fn=gnmt_residual_fn)
        return decoder_cell

    # def _bi_dir_rnn(self):
    #     """
    #     双向RNN
    #     :return:
    #     """
    #     cell_fw = rnn.LSTMCell(self.hidden_unit)
    #     cell_bw = rnn.LSTMCell(self.hidden_unit)
    #     if self.dropout_rate is not None:
    #         cell_bw = rnn.DropoutWrapper(cell_bw, output_keep_prob=self.dropout_rate)
    #         cell_fw = rnn.DropoutWrapper(cell_fw, output_keep_prob=self.dropout_rate)
    #     return cell_fw, cell_bw

    def blstm_layer(self,input):
        """

        :return:
        """
        with tf.variable_scope('rnn_layer'):
            cell_fw = [self.get_rnn_cell() for _ in range(self.num_layers)]
            cell_bw = [self.get_rnn_cell() for _ in range(self.num_layers)]
            # if self.num_layers > 1:
            #     cell_fw = rnn.MultiRNNCell([cell_fw] * self.num_layers, state_is_tuple=True)
            #     cell_bw = rnn.MultiRNNCell([cell_bw] * self.num_layers, state_is_tuple=True)

            rnn_output, _,_ = \
                stack_bidirectional_dynamic_rnn(cell_fw, cell_bw, input,
                                                sequence_length=self.lengths, dtype=tf.float32)
            outputs = tf.concat(rnn_output, axis=2)
        return outputs

    # def project_bilstm_layer(self, lstm_outputs, name=None):
    #     """
    #     hidden layer between lstm layer and logits
    #     :param lstm_outputs: [batch_size, num_steps, emb_size]
    #     :return: [batch_size, num_steps, num_tags]
    #     """
    #     with tf.variable_scope("project" if not name else name):
    #         with tf.variable_scope("hidden"):
    #             W = tf.get_variable("W", shape=[self.hidden_unit * 2, self.hidden_unit],
    #                                 dtype=tf.float32, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=10))
    #
    #             b = tf.get_variable("b", shape=[self.hidden_unit], dtype=tf.float32,
    #                                 initializer=tf.zeros_initializer())
    #             output = tf.reshape(lstm_outputs, shape=[-1, self.hidden_unit * 2])
    #             hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))
    #
    #         # project to score of tags
    #         with tf.variable_scope("logits"):
    #             W = tf.get_variable("W", shape=[self.hidden_unit, self.num_labels],
    #                                 dtype=tf.float32, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=10))
    #
    #             b = tf.get_variable("b", shape=[self.num_labels], dtype=tf.float32,
    #                                 initializer=tf.zeros_initializer())
    #
    #             pred = tf.nn.xw_plus_b(hidden, W, b)
    #         return tf.reshape(pred, [-1, self.seq_length, self.num_labels])

    # def project_crf_layer(self, name=None):
    #     """
    #     hidden layer between input layer and logits
    #     :param lstm_outputs: [batch_size, num_steps, emb_size]
    #     :return: [batch_size, num_steps, num_tags]
    #     """
    #     with tf.variable_scope("project" if not name else name):
    #         with tf.variable_scope("logits"):
    #             W = tf.get_variable("W", shape=[self.embedding_dims, self.num_labels],
    #                                 dtype=tf.float32, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=10))
    #
    #             b = tf.get_variable("b", shape=[self.num_labels], dtype=tf.float32,
    #                                 initializer=tf.zeros_initializer())
    #             output = tf.reshape(self.embedded_chars,
    #                                 shape=[-1, self.embedding_dims])  # [batch_size, embedding_dims]
    #             pred = tf.tanh(tf.nn.xw_plus_b(output, W, b))
    #         return tf.reshape(pred, [-1, self.seq_length, self.num_labels])

    def crf_layer(self, logits):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss"):
            trans = tf.get_variable(
                "transitions",
                shape=[self.num_labels, self.num_labels],
                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=10))
            if self.labels is None:
                return None, trans
            else:
                log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
                    inputs=logits,
                    tag_indices=self.labels,
                    transition_params=trans,
                    sequence_lengths=self.lengths)
                return tf.reduce_mean(-log_likelihood), trans