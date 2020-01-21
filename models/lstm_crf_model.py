import tensorflow as tf
import numpy as np
import common_utils
import optimization
from tf_metrics import precision, recall, f1
from models.utils import dice_dsc_loss
from models.layers.lstm_crf_layers import BLSTM_CRF
from models.layers.lstm_layers import BLSTM
from models.layers.crf_layer import CRF



logger = common_utils.set_logger('NER Training...')


class lstm_crf(tf.keras.models.Model):
    def __init__(self,params:dict):
        super().__init__()
        self.char_embedding = tf.Variable(np.load(params["embedding_path"]),dtype=tf.float32,name='input_char_embedding')
        self.input_dropout = tf.keras.layers.Dropout(params['dropout_prob'])
        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(params['rnn_units'],return_sequences=True))
        self.projection = tf.keras.layers.Dense(params['slot_size'],name='output_slot')
        self.lstm_dropout = tf.keras.layers.Dropout(params['dropout_prob'])

    def call(self, inputs, training=False):
        if inputs.dtype != tf.int32:
            inputs = tf.cast(inputs,tf.int32)
        mask = tf.sign(inputs)
        input_x = tf.nn.embedding_lookup(self.char_embedding,inputs,name="input_embedding_lookup")
        input_x = self.input_dropout(input_x,training=training)
        input_x = self.bilstm(input_x,mask=mask)
        output_x = self.lstm_dropout(input_x,training=training)
        output_projection = self.projection(output_x)
        return output_projection

class BILSTMCRF(object):
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

    def __call__(self,input_ids=None,input_word_ids=None,labels=None,text_length_list=None,is_training=True,is_testing=False):
        input_char_embeddings = tf.nn.embedding_lookup(self.char_embedding, input_ids)
        input_word_embeddings = tf.nn.embedding_lookup(self.word_embedding,input_word_ids)
        # input_word_embeddings = tf.layers.dense(input_word_embeddings,300,activation=tf.nn.relu)
        input_embeddings = input_char_embeddings + input_word_embeddings
        input_embeddings = tf.layers.dropout(input_embeddings, rate=0.5, training=is_training)
        lstm_layer = BLSTM(input_embeddings,self.rnn_size,self.num_layers,1.-self.dropout_rate,lengths=text_length_list,is_training=is_training)
        lstm_output = lstm_layer.blstm_layer(input_embeddings)
        lstm_project = tf.layers.dense(lstm_output, self.num_labels)
        weight = tf.sequence_mask(text_length_list, dtype=tf.float32, name="mask")
        crf_layer = CRF(self.num_labels, labels, text_length_list)
        loss, trans = crf_layer.crf_layer(lstm_project)
        pred_ids = crf_layer.crf_decoding(lstm_project, trans)
        if not is_testing:
            # dice_loss = dice_dsc_loss(lstm_project,labels,weight)
            # lstm_crf_model = BLSTM_CRF(input_embeddings,self.hidden_units,self.rnn_size,self.num_layers,self.dropout_rate,self.num_labels,labels,text_length_list,is_training)
            # final_loss = loss+dice_loss
            return loss, trans, pred_ids, weight
        else:
            return pred_ids
        # corr_target_id_cnt = tf.cast(tf.reduce_sum(
        #     tf.cast(tf.equal(tf.cast(labels, tf.float32), tf.cast(pred_ids, tf.float32)),
        #             tf.float32) * weight), tf.int32)
        # ans_accuracy = corr_target_id_cnt / tf.reduce_sum(text_length_list)
        # # one_hot_labels = tf.one_hot(labels, depth=self.num_labels, dtype=tf.float32)
        # log_probs = tf.nn.log_softmax(logits, axis=-1)
        # per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        # loss = tf.reduce_mean(per_example_loss)
        # return final_loss,trans,pred_ids,weight


def model_fn_builder(args):
    def model_fn(features, labels, mode, params):
        logger.info("*** Features ***")
        if isinstance(features, dict):
            features = features['words'],features['words_seq'], features['text_length']
        input_ids,input_word_ids,text_length_list = features
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        tag_model = BILSTMCRF(params)
        # loss,trans,pred_ids,weight= tag_model(input_ids,input_word_ids,labels,text_length_list,is_training)

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
            loss, trans, pred_ids, weight = tag_model(input_ids, input_word_ids, labels, text_length_list, is_training)
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
            loss, trans, pred_ids, weight = tag_model(input_ids, input_word_ids, labels, text_length_list, is_training)
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
            pred_ids = tag_model(input_ids, input_word_ids, labels, text_length_list, is_training,True)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=pred_ids
            )
        return output_spec
    return model_fn
