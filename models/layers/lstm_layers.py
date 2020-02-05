import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.rnn.python.ops.rnn import stack_bidirectional_dynamic_rnn
from models.layers.general_layers import get_rnn_cell

class BLSTM(object):
    def __init__(self, embedded_chars=None,rnn_size=None, num_layers=None, dropout_rate=None,
                 lengths=None, is_training=None):
        """
        BLSTM网络
        """
        # self.hidden_unit = hidden_unit
        # 保留概率
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

        self.lengths = lengths
        if embedded_chars is not None:
            self.embedding_dims = embedded_chars.shape[-1].value
        self.is_training = is_training
        self.rnn_size = rnn_size

    # def add_blstm_layer(self):
    #     """
    #     blstm-crf网络
    #     :return:
    #     """
    #     # blstm
    #     lstm_output = self.blstm_layer(self.embedded_chars)
    #     # # project
    #     # logits = tf.layers.dense(lstm_output, self.num_labels)
    #     # # crf
    #     # loss, trans = self.crf_layer(logits)
    #     # # CRF decode, pred_ids 是一条最大概率的标注路径
    #     # pred_ids, _ = crf.crf_decode(potentials=logits, transition_params=trans, sequence_length=self.lengths)
    #
    #
    #
    #     # acc = tf.metrics.accuracy(self.labels, pred_ids, weights)
    #     # def metric_fn(label_ids, pred_ids):
    #     #     return precision(label_ids, pred_ids, self.num_labels),recall(label_ids, pred_ids, self.num_labels),f1(label_ids, pred_ids, self.num_labels)
    #     # precisoin_score,recall_score,f1_score = metric_fn(self.labels,pred_ids)
    #     return lstm_output
    #     # return logits

    def blstm_layer(self,input):
        """

        :return:
        """
        with tf.variable_scope('rnn_layer'):
            cell_fw = [get_rnn_cell(self.rnn_size,self.dropout_rate) for _ in range(self.num_layers)]
            cell_bw = [get_rnn_cell(self.rnn_size,self.dropout_rate) for _ in range(self.num_layers)]
            # if self.num_layers > 1:
            #     cell_fw = rnn.MultiRNNCell([cell_fw] * self.num_layers, state_is_tuple=True)
            #     cell_bw = rnn.MultiRNNCell([cell_bw] * self.num_layers, state_is_tuple=True)

            rnn_output, _,_ = \
                stack_bidirectional_dynamic_rnn(cell_fw, cell_bw, input,
                                                sequence_length=self.lengths, dtype=tf.float32)
            outputs = tf.concat(rnn_output, axis=2)
            outputs = tf.layers.dropout(outputs,1.-self.dropout_rate,training=self.is_training)
        return outputs
