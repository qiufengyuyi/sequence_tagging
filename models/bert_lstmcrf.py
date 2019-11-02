import tensorflow as tf
import numpy as np
import common_utils
import optimization
from tf_metrics import precision, recall, f1
from bert import modeling
from models.layers.lstm_crf_layers import BLSTM_CRF
logger = common_utils.set_logger('NER Training...')

class bertLSTMCRF(object):
    def __init__(self,params,bert_config):
        self.dropout_rate = params["dropout_prob"]
        self.num_labels = params["num_labels"]
        self.rnn_size = params["rnn_size"]
        self.num_layers = params["num_layers"]
        self.hidden_units = params["hidden_units"]
        self.bert_config = bert_config

    def __call__(self,input_ids,labels,text_length_list,is_training):

        bert_model = modeling.BertModel(
            config=self.bert_config,
            is_training=is_training,
            input_ids=input_ids,
            text_length=text_length_list,
            use_one_hot_embeddings=False
        )
        bert_embedding = bert_model.get_sequence_output()
        lstm_crf_model = BLSTM_CRF(bert_embedding,self.hidden_units,self.rnn_size,self.num_layers,self.dropout_rate,self.num_labels,labels,text_length_list,is_training)
        loss, logits, trans, pred_ids = lstm_crf_model.add_blstm_crf_layer()
        weight = tf.sequence_mask(text_length_list, dtype=tf.float32, name="mask")
        return loss, logits, trans, pred_ids, weight

def bert_model_fn_builder(bert_config_file,init_checkpoints,args):
    def model_fn(features, labels, mode, params):
        logger.info("*** Features ***")
        if isinstance(features, dict):
            features = features['words'],features['text_length']
        input_ids,text_length_list = features
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        bert_config = modeling.BertConfig.from_json_file(bert_config_file)
        tag_model = bertLSTMCRF(params,bert_config)
        loss,logits,trans,pred_ids,weight= tag_model(input_ids,labels,text_length_list,is_training)

        # def metric_fn(label_ids, pred_ids):
        #     return {
        #         'precision': precision(label_ids, pred_ids, params["num_labels"]),
        #         'recall': recall(label_ids, pred_ids, params["num_labels"]),
        #         'f1': f1(label_ids, pred_ids, params["num_labels"])
        #     }
        #
        # eval_metrics = metric_fn(labels, pred_ids)
        tvars = tf.trainable_variables()
        # 加载BERT模型
        if init_checkpoints:
            (assignment_map, initialized_variable_names) = \
                modeling.get_assignment_map_from_checkpoint(tvars,
                                                            init_checkpoints)
            tf.train.init_from_checkpoint(init_checkpoints, assignment_map)
        output_spec = None
        # f1_score_val, f1_update_op_val = f1(labels=labels, predictions=pred_ids, num_classes=params["num_labels"],
        #                                     weights=weight)

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
            eval_loss = tf.metrics.mean_squared_error(labels=labels, predictions=pred_ids,weights=weight)
            eval_metric_ops = {
                "eval_loss":eval_loss,
                "f1_score":(f1_score_val,f1_update_op_val)}
            # eval_hook_dict = {"f1":f1_score_val,"loss":loss}

            # eval_logging_hook = tf.train.LoggingTensorHook(
            #     at_end=True,every_n_iter=args.print_log_steps)
            output_spec = tf.estimator.EstimatorSpec(
                eval_metric_ops=eval_metric_ops,
                mode=mode,
                loss=loss
            )
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=pred_ids
            )
        return output_spec
    return model_fn