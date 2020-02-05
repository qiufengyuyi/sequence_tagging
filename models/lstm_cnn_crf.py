import tensorflow as tf
import numpy as np
import common_utils
import optimization
from models.tf_metrics import f1
import sys
from models.layers.crf_layers import CRF
from models.layers.lstm_layers import BLSTM
from models.layers.cnn_layers import CNNLAYER


logger = common_utils.set_logger('CNN NER Training...')

class BILSTMCNNCRF(object):
    def __init__(self,params:dict):
        self.char_embedding = tf.Variable(np.load(params["embedding_path"]), dtype=tf.float32,
                                          name='input_char_embedding')
        self.word_embedding = tf.Variable(np.load(params["word_embedding_path"]),dtype=tf.float32,name="input_word_embedding")
        # 丢弃概率
        self.dropout_rate = params["dropout_prob"]
        self.num_labels = params["num_labels"]
        self.rnn_size = params["rnn_size"]
        self.num_layers = params["num_layers"]
        self.hidden_units = params["hidden_units"]
        self.kernel_nums = params["kernel_nums"]
        self.kernel_size = params["kernel_size"]

    def __call__(self, input_ids=None,input_word_ids=None,labels=None,text_length_list=None,is_training=True):
        input_char_embeddings = tf.nn.embedding_lookup(self.char_embedding, input_ids)
        input_word_embeddings = tf.nn.embedding_lookup(self.word_embedding,input_word_ids)
        input_embeddings = input_char_embeddings + input_word_embeddings
        input_embeddings = tf.layers.dropout(input_embeddings, rate=self.dropout_rate, training=is_training)
        lstm_layer = BLSTM(input_embeddings, self.rnn_size, self.num_layers, 1.-self.dropout_rate,
                           lengths=text_length_list, is_training=is_training)
        lstm_output = lstm_layer.blstm_layer(input_embeddings)
        cnn_layer = CNNLAYER(self.kernel_size,self.kernel_nums,self.dropout_rate,self.hidden_units,is_training=is_training)
        cnn_output = cnn_layer.add_conv(lstm_output)
        # print("----------------cnn_output:{}".format(cnn_output))
        # tf.print("----------------cnn_output:",cnn_output,output_stream=sys.stderr)
        # output_merge = tf.concat([lstm_output,cnn_output],axis=-1)
        # print("----------------output_merge:{}".format(output_merge))
        crf_inputs = tf.layers.dense(cnn_output,self.num_labels)
        # print("----------------crf_inputs:{}".format(crf_inputs))
        crf_layer = CRF(self.num_labels, labels, text_length_list)
        loss, trans = crf_layer.crf_layer(crf_inputs)
        pred_ids = crf_layer.crf_decoding(crf_inputs,trans)
        weight = tf.sequence_mask(text_length_list, dtype=tf.float32, name="mask")
        return loss, trans, pred_ids, weight


def cnn_model_fn_builder(args):
    def model_fn(features, labels, mode, params):
        logger.info("*** Features ***")
        if isinstance(features, dict):
            features = features['words'],features['words_seq'], features['text_length']
        input_ids,input_word_ids,text_length_list = features
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        tag_model = BILSTMCNNCRF(params)
        loss,trans,pred_ids,weight= tag_model(input_ids,input_word_ids,labels,text_length_list,is_training)

        # def metric_fn(label_ids, pred_ids):
        #     return {
        #         'precision': precision(label_ids, pred_ids, params["num_labels"]),
        #         'recall': recall(label_ids, pred_ids, params["num_labels"]),
        #         'f1': f1(label_ids, pred_ids, params["num_labels"])
        #     }
        #
        # eval_metrics = metric_fn(labels, pred_ids)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(loss,args.lr, params["decay_steps"],args.clip_norm)
            hook_dict = {}
            # precision_score, precision_update_op = precision(labels=labels, predictions=pred_ids,
            #                                                  num_classes=params["num_labels"], weights=weight)
            #
            # recall_score, recall_update_op = recall(labels=labels,
            #                                         predictions=pred_ids, num_classes=params["num_labels"],
            #                                         weights=weight)
            hook_dict['loss'] = loss
            hook_dict['global_steps'] = tf.train.get_or_create_global_step()
            logging_hook = tf.train.LoggingTensorHook(
                hook_dict, every_n_iter=args.print_log_steps)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                training_hooks=[logging_hook])

        elif mode == tf.estimator.ModeKeys.EVAL:
            # pred_ids = tf.argmax(logits, axis=-1, output_type=tf.int32)
            # weight = tf.sequence_mask(text_length_list)
            # precision_score, precision_update_op = precision(labels=labels,predictions=pred_ids,num_classes=params["num_labels"],weights=weight)
            #
            # recall_score, recall_update_op =recall(labels=labels,
            #                                              predictions=pred_ids,num_classes=params["num_labels"],weights=weight)
            f1_score_val,f1_update_op_val = f1(labels=labels,predictions=pred_ids,num_classes=params["num_labels"],weights=weight)
            # acc_score_val,acc_score_op_val = tf.metrics.accuracy(labels=labels,predictions=pred_ids,weights=weight)
            eval_metric_ops = {
                "f1":(f1_score_val,f1_update_op_val)}
            eval_hook_dict = {"f1":f1_score_val,"loss":loss}

            eval_logging_hook = tf.train.LoggingTensorHook(
                eval_hook_dict,at_end=True,every_n_iter=args.print_log_steps)
            output_spec = tf.estimator.EstimatorSpec(
                eval_metric_ops=eval_metric_ops,
                mode=mode,
                loss=loss,
                evaluation_hooks=[eval_logging_hook]
            )
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=pred_ids
            )
        return output_spec
    return model_fn
