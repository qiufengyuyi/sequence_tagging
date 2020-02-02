import os
import numpy as np
import time
import logging
from common_utils import set_logger
import tensorflow as tf
from models.lstm_crf_model import model_fn_builder
from models.bert_lstmcrf import bert_model_fn_builder
from models.lstm_cnn_crf import cnn_model_fn_builder
from models.lstm_only import lstm_model_fn_builder
from models.bert_mrc import bert_mrc_model_fn_builder
from data_processing.data_utils import *
from data_processing.basic_prepare_data import BaseDataPreparing
from data_processing.bert_prepare_data import bertPrepareData
from data_processing.bert_mrc_prepare_data import bertMRCPrepareData
from configs.base_config import config
from configs.bert_config import bert_config
from configs.bert_mrc_config import bert_mrc_config
from bert import modeling
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

logger = set_logger("[run training]")
# logger = logging.getLogger('train')
# logger.setLevel(logging.INFO)


def serving_input_receiver_fn():
    """Serving input_fn that builds features from placeholders

    Returns
    -------
    tf.estimator.export.ServingInputReceiver
    """
    words = tf.placeholder(dtype=tf.int32, shape=[None, None], name='words')
    nwords = tf.placeholder(dtype=tf.int32, shape=[None], name='text_length')
    words_seq = tf.placeholder(dtype=tf.int32, shape=[None,None], name='words_seq')
    receiver_tensors = {'words': words, 'text_length': nwords,'words_seq':words_seq}
    features = {'words': words, 'text_length': nwords,'words_seq':words_seq}
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

def bert_serving_input_receiver_fn():
    """Serving input_fn that builds features from placeholders

    Returns
    -------
    tf.estimator.export.ServingInputReceiver
    """
    words = tf.placeholder(dtype=tf.int32, shape=[None, None], name='words')
    nwords = tf.placeholder(dtype=tf.int32, shape=[None], name='text_length')
    receiver_tensors = {'words': words, 'text_length': nwords}
    features = {'words': words, 'text_length': nwords}
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

def bert_mrc_serving_input_receiver_fn():
    # features['words'],features['text_length'],features['query_length'],features['token_type_ids']
    words = tf.placeholder(dtype=tf.int32, shape=[None, None], name='words')
    nwords = tf.placeholder(dtype=tf.int32, shape=[None], name='text_length')
    query_lengths = tf.placeholder(dtype=tf.int32,shape=[None],name="query_length")
    token_type_ids = tf.placeholder(dtype=tf.int32,shape=[None,None],name="token_type_ids")
    receiver_tensors = {'words': words, 'text_length': nwords,'query_length':query_lengths,'token_type_ids':token_type_ids}
    features = {'words': words, 'text_length': nwords,'query_length':query_lengths,'token_type_ids':token_type_ids}
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

class RestoreCheckpointHook(tf.train.SessionRunHook):
    def __init__(self,init_checkpoints
                 ):
        tf.logging.info("Create RestoreCheckpointHook.")
        self.init_checkpoints = init_checkpoints

    def begin(self):
        tvars = tf.trainable_variables()
        (assignment_map, initialized_variable_names
         ) = modeling.get_assignment_map_from_checkpoint(
            tvars, self.init_checkpoints)
        tf.train.init_from_checkpoint(
            self.init_checkpoints, assignment_map)

        self.saver = tf.train.Saver(tvars)

    def after_create_session(self, session, coord):

        pass

    def before_run(self, run_context):
        return None

    def after_run(self, run_context, run_values):
        pass

    def end(self, session):
        pass

def run_bert(args):
    vocab_file_path = os.path.join(bert_config.get("bert_pretrained_model_path"), bert_config.get("vocab_file"))
    bert_config_file = os.path.join(bert_config.get("bert_pretrained_model_path"), config.get("bert_config_path"))
    slot_file = os.path.join(bert_config.get("slot_list_root_path"), bert_config.get("bert_slot_file_name"))
    data_loader = bertPrepareData(vocab_file_path,slot_file,bert_config,bert_config_file,384,True,False)
    print(data_loader.train_valid_split_data_path)
    if data_loader.train_samples_nums % args.train_batch_size !=0:
        each_epoch_steps = int(data_loader.train_samples_nums / args.train_batch_size) + 1
    else:
        each_epoch_steps = int(data_loader.train_samples_nums / args.train_batch_size)
    # each_epoch_steps = int(data_loader.train_samples_nums/args.train_batch_size)+1
    logger.info('*****train_set sample nums:{}'.format(data_loader.train_samples_nums))
    logger.info('*****train each epoch steps:{}'.format(each_epoch_steps))
    train_steps_nums = each_epoch_steps*args.epochs
    logger.info('*****train_total_steps:{}'.format(train_steps_nums))
    decay_steps = args.decay_epoch*each_epoch_steps
    logger.info('*****train decay steps:{}'.format(decay_steps))
    # dropout_prob是丢弃概率
    params = {"dropout_prob": args.dropout_prob, "num_labels": data_loader.slot_label_size,
              "rnn_size": args.rnn_units, "num_layers": args.num_layers, "hidden_units": args.hidden_units,
              "decay_steps":decay_steps}
    # dist_strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=args.gpu_nums)
    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth = True
    #"bert_ce_model_dir"
    run_config = tf.estimator.RunConfig(
        model_dir=bert_config.get(args.model_checkpoint_dir),
        save_summary_steps=each_epoch_steps,
        save_checkpoints_steps=each_epoch_steps,
        session_config=config_tf,
        keep_checkpoint_max=2,
        # train_distribute=dist_strategy

    )

    bert_init_checkpoints = os.path.join(bert_config.get("bert_pretrained_model_path"),bert_config.get("bert_init_checkpoints"))
    model_fn = bert_model_fn_builder(bert_config_file,bert_init_checkpoints,args)
    estimator = tf.estimator.Estimator(
        model_fn,
        params=params,
        config=run_config)
    # train_hook_one = RestoreCheckpointHook(bert_init_checkpoints)
    early_stopping_hook = tf.contrib.estimator.stop_if_no_decrease_hook(
        estimator=estimator,
        metric_name='loss',
        max_steps_without_decrease=args.tolerant_steps,
        eval_dir=None,
        min_steps=0,
        run_every_secs=None,
        run_every_steps=args.run_hook_steps)
    if args.do_train:
        # train_input_fn = lambda: data_loader.create_dataset(is_training=True,is_testing=False, args=args)
        # eval_input_fn = lambda: data_loader.create_dataset(is_training=False,is_testing=False,args=args)
        train_X,train_Y = np.load(data_loader.train_X_path,allow_pickle=True),np.load(data_loader.train_Y_path,allow_pickle=True)
        train_input_fn = lambda :input_bert_fn(train_X,train_Y,is_training=True,args=args)
        eval_X,eval_Y = np.load(data_loader.valid_X_path,allow_pickle=True),np.load(data_loader.valid_Y_path,allow_pickle=True)

        eval_input_fn = lambda: input_bert_fn(eval_X,eval_Y,is_training=False,args=args)
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=train_steps_nums,
                                            hooks=[early_stopping_hook]
                                            )
        exporter = tf.estimator.BestExporter(exports_to_keep=1,serving_input_receiver_fn=bert_serving_input_receiver_fn)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,exporters=[exporter],throttle_secs=0)
        # for _ in range(args.epochs):

        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        #"bert_ce_model_pb"
        estimator.export_saved_model(bert_config.get(args.model_pb_dir), bert_serving_input_receiver_fn)

def run_bert_mrc(args):
    vocab_file_path = os.path.join(bert_mrc_config.get("bert_pretrained_model_path"), bert_mrc_config.get("vocab_file"))
    bert_config_file = os.path.join(bert_mrc_config.get("bert_pretrained_model_path"), bert_mrc_config.get("bert_config_path"))
    slot_file = os.path.join(bert_mrc_config.get("slot_list_root_path"), bert_mrc_config.get("bert_slot_file_name"))
    data_loader = bertMRCPrepareData(vocab_file_path,slot_file,bert_mrc_config,bert_config_file,512,True,False)
    print(data_loader.train_valid_split_data_path)
    if data_loader.train_samples_nums % args.train_batch_size !=0:
        each_epoch_steps = int(data_loader.train_samples_nums / args.train_batch_size) + 1
    else:
        each_epoch_steps = int(data_loader.train_samples_nums / args.train_batch_size)
    # each_epoch_steps = int(data_loader.train_samples_nums/args.train_batch_size)+1
    logger.info('*****train_set sample nums:{}'.format(data_loader.train_samples_nums))
    logger.info('*****train each epoch steps:{}'.format(each_epoch_steps))
    train_steps_nums = each_epoch_steps*args.epochs
    logger.info('*****train_total_steps:{}'.format(train_steps_nums))
    decay_steps = args.decay_epoch*each_epoch_steps
    logger.info('*****train decay steps:{}'.format(decay_steps))
    # dropout_prob是丢弃概率
    params = {"dropout_prob": args.dropout_prob, "num_labels": data_loader.slot_label_size,
              "rnn_size": args.rnn_units, "num_layers": args.num_layers, "hidden_units": args.hidden_units,
              "decay_steps":decay_steps}
    # dist_strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=args.gpu_nums)
    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth = True
    #"bert_ce_model_dir"
    run_config = tf.estimator.RunConfig(
        model_dir=bert_mrc_config.get(args.model_checkpoint_dir),
        save_summary_steps=each_epoch_steps,
        save_checkpoints_steps=each_epoch_steps,
        session_config=config_tf,
        keep_checkpoint_max=2,
        # train_distribute=dist_strategy

    )

    bert_init_checkpoints = os.path.join(bert_mrc_config.get("bert_pretrained_model_path"),bert_mrc_config.get("bert_init_checkpoints"))
    model_fn = bert_mrc_model_fn_builder(bert_config_file,bert_init_checkpoints,args)
    estimator = tf.estimator.Estimator(
        model_fn,
        params=params,
        config=run_config)
    # train_hook_one = RestoreCheckpointHook(bert_init_checkpoints)
    early_stopping_hook = tf.contrib.estimator.stop_if_no_decrease_hook(
        estimator=estimator,
        metric_name='loss',
        max_steps_without_decrease=args.tolerant_steps,
        eval_dir=None,
        min_steps=0,
        run_every_secs=None,
        run_every_steps=args.run_hook_steps)
    if args.do_train:
        # train_input_fn = lambda: data_loader.create_dataset(is_training=True,is_testing=False, args=args)
        # eval_input_fn = lambda: data_loader.create_dataset(is_training=False,is_testing=False,args=args)
        train_X = np.load(data_loader.train_X_path,allow_pickle=True)
        train_start_Y = np.load(data_loader.train_start_Y_path,allow_pickle=True)
        train_end_Y = np.load(data_loader.train_end_Y_path, allow_pickle=True)
        train_query_lens = np.load(data_loader.train_query_len_path,allow_pickle=True)
        train_token_type_ids = np.load(data_loader.train_token_type_ids_path,allow_pickle=True)
        # input_Xs,start_Ys,end_Ys,token_type_ids,query_lens
        train_input_fn = lambda :input_bert_mrc_fn(train_X,train_start_Y,train_end_Y,train_token_type_ids,train_query_lens,is_training=True,args=args)
        eval_X = np.load(data_loader.valid_X_path,allow_pickle=True)
        eval_start_Y = np.load(data_loader.valid_start_Y_path,allow_pickle=True)
        eval_end_Y = np.load(data_loader.valid_end_Y_path, allow_pickle=True)
        eval_query_lens = np.load(data_loader.valid_query_len_path,allow_pickle=True)
        eval_token_type_ids = np.load(data_loader.valid_token_type_ids_path,allow_pickle=True)

        eval_input_fn = lambda: input_bert_mrc_fn(eval_X,eval_start_Y,eval_end_Y,eval_token_type_ids,eval_query_lens,is_training=False,args=args)
        exporter = tf.estimator.BestExporter(exports_to_keep=1,
                                             serving_input_receiver_fn=bert_mrc_serving_input_receiver_fn)
        for i in range(args.epochs):
            train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=train_steps_nums
                                                )

            eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,exporters=[exporter],throttle_secs=0)
            # for _ in range(args.epochs):

            tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        #"bert_ce_model_pb"
        estimator.export_saved_model(bert_mrc_config.get(args.model_pb_dir), bert_mrc_serving_input_receiver_fn)


def run_train(args):
    model_type = args.model_type
    model_base_dir = config.get(args.model_checkpoint_dir)
    pb_model_dir = config.get(args.model_pb_dir)
    print(model_base_dir)
    print(pb_model_dir)
    vocab_file_path = os.path.join(config.get("embedding_dir"), config.get("vocab_file"))
    pretrained_embedding_path = os.path.join(config.get("embedding_dir"), config.get("embedding_file_name"))
    input_embedding_path = os.path.join(config.get("embedding_dir"), config.get("input_embedding_file"))
    input_word_embedding_path = os.path.join(config.get("embedding_dir"), config.get("input_word_embedding_file"))
    slot_file = os.path.join(config.get("slot_list_root_path"), config.get("slot_file_name"))
    t0 = time.time()
    print(args.gen_new_data)
    data_loader = BaseDataPreparing(vocab_file_path, slot_file,config, pretrained_embedding_path, input_embedding_path,input_word_embedding_path,
                                    load_w2v_embedding=True,load_word_embedding=True,gen_new_data=args.gen_new_data,is_inference=False)
    if data_loader.train_samples_nums % args.train_batch_size !=0:
        each_epoch_steps = int(data_loader.train_samples_nums / args.train_batch_size) + 1
    else:
        each_epoch_steps = int(data_loader.train_samples_nums / args.train_batch_size)
    # each_epoch_steps = int(data_loader.train_samples_nums/args.train_batch_size)+1
    logger.info('*****train_set sample nums:{}'.format(data_loader.train_samples_nums))
    logger.info('*****train each epoch steps:{}'.format(each_epoch_steps))
    train_steps_nums = each_epoch_steps*args.epochs
    logger.info('*****train_total_steps:{}'.format(train_steps_nums))
    decay_steps = args.decay_epoch*each_epoch_steps
    logger.info('*****train decay steps:{}'.format(decay_steps))
    params = {"dropout_prob": args.dropout_prob, "num_labels": data_loader.slot_label_size,
              "rnn_size": args.rnn_units, "num_layers": args.num_layers, "hidden_units": args.hidden_units,
              "embedding_path":input_embedding_path,"decay_steps":decay_steps,"word_embedding_path":input_word_embedding_path}

    # params = {"dropout_prob": args.dropout_prob, "num_labels": data_loader.slot_label_size,
    #           "rnn_size": args.rnn_units, "num_layers": args.num_layers,
    #           "embedding_path": input_embedding_path, "decay_steps": decay_steps,
    #           "attention_size": args.attention_size,
    #           "num_header": args.num_header, "slot_list": data_loader.slot_list,
    #           "label_embedding_size": args.label_embedding_size,"word_embedding_path":input_word_embedding_path}

    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth = True
    run_config = tf.estimator.RunConfig(
        model_dir=model_base_dir,
        save_summary_steps=each_epoch_steps,
        save_checkpoints_steps=each_epoch_steps,
        session_config=config_tf,
        keep_checkpoint_max=1

    )
    if model_type == "lstm_only":
        model_fn = lstm_model_fn_builder(args)
    else:
        model_fn = model_fn_builder(args)
    # model_fn = lan_model_fn_builder(args)
    estimator = tf.estimator.Estimator(
        model_fn,
        params=params,
        config=run_config)

    early_stopping_hook = tf.contrib.estimator.stop_if_no_decrease_hook(
        estimator=estimator,
        metric_name='loss',
        max_steps_without_decrease=args.tolerant_steps,
        eval_dir=None,
        min_steps=0,
        run_every_secs=None,
        run_every_steps=args.run_hook_steps)

    if args.do_train:
        # train_input_fn = lambda: data_loader.create_dataset(is_training=True,is_testing=False, args=args)
        # eval_input_fn = lambda: data_loader.create_dataset(is_training=False,is_testing=False,args=args)
        train_X,train_Y = np.load(data_loader.train_X_path,allow_pickle=True),np.load(data_loader.train_Y_path,allow_pickle=True)
        train_X_word = np.load(data_loader.train_word_path,allow_pickle=True)
        train_rand_i=np.random.permutation(len(train_X))
        train_X = train_X[train_rand_i]
        train_Y = train_Y[train_rand_i]
        train_X_word = train_X_word[train_rand_i]
        train_input_fn = lambda :input_fn(train_X,train_X_word,train_Y,is_training=True,args=args)
        eval_X,eval_Y = np.load(data_loader.valid_X_path,allow_pickle=True),np.load(data_loader.valid_Y_path,allow_pickle=True)
        eval_X_word = np.load(data_loader.valid_word_path,allow_pickle=True)
        eval_input_fn = lambda: input_fn(eval_X,eval_X_word,eval_Y,is_training=False,args=args)
        exporter = tf.estimator.BestExporter(exports_to_keep=1, serving_input_receiver_fn=serving_input_receiver_fn)
        for i in range(args.epochs):
            train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=train_steps_nums,
                                                hooks=[early_stopping_hook]
                                                )
            eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, exporters=[exporter], steps=each_epoch_steps,
                                              throttle_secs=0)
            tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

        estimator.export_saved_model(pb_model_dir, serving_input_receiver_fn)
    # if args.do_test:
    #     test_input_fn = lambda: input_fn(data_loader,is_training=False,is_testing=False, args=args)
    #     result = estimator.predict(input_fn=test_input_fn)
    #     for res in result:
    #         print(res)
        # golden_labels = np.load(data_loader.test_Y_path,allow_pickle=True)
        #
        # tf.logging.info("---------eval f1 score-------:{}".format(f1_score(golden_labels,result,average="micro")))
        # tf.logging.info("---------eval classification report--------\n{}".format(classification_report(golden_labels,result)))

def run_train_cnn(args):
    model_type = args.model_type
    model_base_dir = config.get(args.model_checkpoint_dir)
    pb_model_dir = config.get(args.model_pb_dir)
    print(model_base_dir)
    print(pb_model_dir)
    vocab_file_path = os.path.join(config.get("embedding_dir"), config.get("vocab_file"))
    pretrained_embedding_path = os.path.join(config.get("embedding_dir"), config.get("embedding_file_name"))
    input_embedding_path = os.path.join(config.get("embedding_dir"), config.get("input_embedding_file"))
    input_word_embedding_path = os.path.join(config.get("embedding_dir"), config.get("input_word_embedding_file"))
    slot_file = os.path.join(config.get("slot_list_root_path"), config.get("slot_file_name"))
    t0 = time.time()
    # model_type = args.model_type
    data_loader = BaseDataPreparing(vocab_file_path, slot_file, config, pretrained_embedding_path, input_embedding_path,
                                    input_word_embedding_path,
                                    load_w2v_embedding=True, load_word_embedding=True, gen_new_data=args.gen_new_data,
                                    is_inference=False)
    if data_loader.train_samples_nums % args.train_batch_size !=0:
        each_epoch_steps = int(data_loader.train_samples_nums / args.train_batch_size) + 1
    else:
        each_epoch_steps = int(data_loader.train_samples_nums / args.train_batch_size)
    # each_epoch_steps = int(data_loader.train_samples_nums/args.train_batch_size)+1
    logger.info('*****train_set sample nums:{}'.format(data_loader.train_samples_nums))
    logger.info('*****train each epoch steps:{}'.format(each_epoch_steps))
    train_steps_nums = each_epoch_steps*args.epochs
    logger.info('*****train_total_steps:{}'.format(train_steps_nums))
    decay_steps = args.decay_epoch*each_epoch_steps
    logger.info('*****train decay steps:{}'.format(decay_steps))
    params = {"dropout_prob": args.dropout_prob, "num_labels": data_loader.slot_label_size,
              "rnn_size": args.rnn_units, "num_layers": args.num_layers, "hidden_units": args.hidden_units,
              "embedding_path":input_embedding_path,"decay_steps":decay_steps,"word_embedding_path":input_word_embedding_path,
              "kernel_size":args.kernel_size,"kernel_nums":args.kernel_nums}

    # params = {"dropout_prob": args.dropout_prob, "num_labels": data_loader.slot_label_size,
    #           "rnn_size": args.rnn_units, "num_layers": args.num_layers,
    #           "embedding_path": input_embedding_path, "decay_steps": decay_steps,
    #           "attention_size": args.attention_size,
    #           "num_header": args.num_header, "slot_list": data_loader.slot_list,
    #           "label_embedding_size": args.label_embedding_size,"word_embedding_path":input_word_embedding_path}

    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth = True
    run_config = tf.estimator.RunConfig(
        model_dir=model_base_dir,
        save_summary_steps=each_epoch_steps,
        save_checkpoints_steps=each_epoch_steps,
        session_config=config_tf,
        keep_checkpoint_max=1

    )
    model_fn = cnn_model_fn_builder(args)
    # model_fn = lan_model_fn_builder(args)
    estimator = tf.estimator.Estimator(
        model_fn,
        params=params,
        config=run_config)

    early_stopping_hook = tf.contrib.estimator.stop_if_no_decrease_hook(
        estimator=estimator,
        metric_name='loss',
        max_steps_without_decrease=args.tolerant_steps,
        eval_dir=None,
        min_steps=0,
        run_every_secs=None,
        run_every_steps=args.run_hook_steps)

    if args.do_train:
        # train_input_fn = lambda: data_loader.create_dataset(is_training=True,is_testing=False, args=args)
        # eval_input_fn = lambda: data_loader.create_dataset(is_training=False,is_testing=False,args=args)
        train_X,train_Y = np.load(data_loader.train_X_path,allow_pickle=True),np.load(data_loader.train_Y_path,allow_pickle=True)
        train_X_word = np.load(data_loader.train_word_path,allow_pickle=True)
        train_rand_i=np.random.permutation(len(train_X))
        train_X = train_X[train_rand_i]
        train_Y = train_Y[train_rand_i]
        train_X_word = train_X_word[train_rand_i]
        train_input_fn = lambda :input_fn(train_X,train_X_word,train_Y,is_training=True,args=args)
        eval_X,eval_Y = np.load(data_loader.valid_X_path,allow_pickle=True),np.load(data_loader.valid_Y_path,allow_pickle=True)
        eval_X_word = np.load(data_loader.valid_word_path,allow_pickle=True)
        eval_input_fn = lambda: input_fn(eval_X,eval_X_word,eval_Y,is_training=False,args=args)
        exporter = tf.estimator.BestExporter(exports_to_keep=1, serving_input_receiver_fn=serving_input_receiver_fn)
        for i in range(args.epochs):
            train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=train_steps_nums,
                                                        hooks=[early_stopping_hook]
                                                        )
            eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, exporters=[exporter],steps=each_epoch_steps,throttle_secs=0)
            tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        # while True:
        #     try:
        #         train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=train_steps_nums,
        #                                                 hooks=[early_stopping_hook]
        #                                                 )
        #         eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, exporters=[exporter],throttle_secs=0)
        #         tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        #     except Exception as e:
        #         logger.info("----------")
        #         logger.exception(e)
        #         break

        estimator.export_saved_model(pb_model_dir, serving_input_receiver_fn)
