import tensorflow as tf
import numpy as np
import common_utils
import optimization
from models.tf_metrics import precision, recall, f1
from bert import modeling
from models.layers.lstm_crf_layers import BLSTM_CRF
from models.layers.crf_layers import CRF
from models.layers.lstm_layers import BLSTM
logger = common_utils.set_logger('NER Training...')

class bertLSTMCRF(object):
    def __init__(self,params,bert_config):
        # 丢弃概率
        self.dropout_rate = params["dropout_prob"]
        self.num_labels = params["num_labels"]
        self.rnn_size = params["rnn_size"]
        self.num_layers = params["num_layers"]
        self.hidden_units = params["hidden_units"]
        self.bert_config = bert_config

    def __call__(self,input_ids,labels,text_length_list,is_training,is_testing=False):

        bert_model = modeling.BertModel(
            config=self.bert_config,
            is_training=is_training,
            input_ids=input_ids,
            text_length=text_length_list,
            use_one_hot_embeddings=False
        )
        bert_embedding = bert_model.get_sequence_output()
        lstm_layer = BLSTM(None, self.rnn_size, self.num_layers, 1.-self.dropout_rate,
                           lengths=text_length_list, is_training=is_training)
        lstm_output = lstm_layer.blstm_layer(bert_embedding)

        # lstm_crf_model = BLSTM_CRF(bert_embedding,self.hidden_units,self.rnn_size,self.num_layers,self.dropout_rate,self.num_labels,labels,text_length_list,is_training)
        # loss, logits, trans, pred_ids = lstm_crf_model.add_blstm_crf_layer()

        # bert_project = tf.layers.dense(bert_embedding, self.hidden_units,activation=tf.nn.relu)
        # bert_project = tf.layers.dropout(bert_project,rate=self.dropout_rate,training=is_training)
        # bert_project = tf.layers.dense(bert_project, self.num_labels)
        # pred_ids = tf.argmax(bert_project, axis=-1, name="pred_ids")


        crf_input = tf.layers.dense(lstm_output,self.num_labels)
        crf_layer = CRF(self.num_labels, labels, text_length_list)
        loss, trans = crf_layer.crf_layer(crf_input)
        pred_ids = crf_layer.crf_decoding(crf_input, trans)

        weight = tf.sequence_mask(text_length_list, dtype=tf.float32, name="mask")
        # pred_prob = tf.nn.softmax(bert_project, axis=-1, name="pred_probs")

        if not is_testing:
            # one_hot_labels = tf.one_hot(labels, depth=self.num_labels, dtype=tf.float32)
            # log_probs = tf.nn.log_softmax(bert_project, axis=-1)
            # per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            # per_example_loss = per_example_loss * weight
            # loss = tf.reduce_sum(per_example_loss,axis=-1)
            # loss = tf.reduce_mean(loss)


            # loss = tf.reduce_mean(per_example_loss)
            # loss = dice_dsc_loss(bert_project,labels,text_length_list,weight,self.num_labels)
            # loss = focal_dsc_loss(bert_project,labels,text_length_list,weight,self.num_labels)

            return loss, pred_ids, weight
        else:
            return pred_ids


def bert_model_fn_builder(bert_config_file,init_checkpoints,args):
    def model_fn(features, labels, mode, params):
        logger.info("*** Features ***")
        if isinstance(features, dict):
            features = features['words'],features['text_length']
        print(features)
        input_ids,text_length_list = features
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        is_testing = (mode == tf.estimator.ModeKeys.PREDICT)
        bert_config = modeling.BertConfig.from_json_file(bert_config_file)
        tag_model = bertLSTMCRF(params,bert_config)
        if is_testing:
            pred_ids = tag_model(input_ids, labels, text_length_list, is_training,is_testing)
        else:
            loss,pred_ids,weight= tag_model(input_ids,labels,text_length_list,is_training)

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
            # f1_score_val,f1_update_op_val = f1(labels=labels,predictions=pred_ids,num_classes=params["num_labels"],weights=weight,average="macro")
            f1_score_val_micro,f1_update_op_val_micro = f1(labels=labels,predictions=pred_ids,num_classes=params["num_labels"],weights=weight,average="micro")

            # acc_score_val,acc_score_op_val = tf.metrics.accuracy(labels=labels,predictions=pred_ids,weights=weight)
            # eval_loss = tf.metrics.mean_squared_error(labels=labels, predictions=pred_ids,weights=weight)
            eval_metric_ops = {
            "f1_score_micro":(f1_score_val_micro,f1_update_op_val_micro)}
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
