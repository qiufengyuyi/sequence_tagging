import os
import codecs
import gensim
import requests
import numpy as np
import json
import tensorflow as tf
import torch
from torch.utils.data import TensorDataset
SEGURL=""

def seg_from_api(data):
    try:
        datas = {"text": data}
        headers = {'Content-Type': 'application/json'}
        res = requests.post(SEGURL, data=json.dumps(datas), headers=headers)
        text = res.text
        text_dict = json.loads(text)
        return text_dict
    except:
        print("dfdfdf")

def seg(text):
    words = seg_from_api(text)
    word_list = [word.get("word") for word in words]
    return word_list

def read_slots(slot_file_path=None,slot_source_type="file"):
    """
    根据不同的槽位模板文件，生成槽位的label
    :param slot_file_path:
    :param slot_source_type:
    :return:
    """
    slot2id_dict = {}
    id2slot_dict = {}
    if slot_source_type == "file":
        with codecs.open(slot_file_path,'r','utf-8') as fr:
            for i,line in enumerate(fr):
                line=line.strip("\n")
                line = line.strip("\r")
                slot2id_dict[line] = i
                id2slot_dict[i] = line
    return slot2id_dict,id2slot_dict

def gen_char_embedding(pretrained_char_embedding_file=None,gram_dict=None,embedding_dim=300,output_file=None):
    if not os.path.exists(output_file):
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(pretrained_char_embedding_file, binary=False,unicode_errors='ignore')
        text_wordvec = np.zeros((len(gram_dict), embedding_dim))
        print("gen_word2vec.....")
        count = 0
        for word, word_index in gram_dict.items():
            count += 1
            if count % 500 == 0:
                print("count:{}.......".format(count))
            try:
                word_vec = word2vec[word]
                text_wordvec[word_index] = word_vec
            except:
                print("exception:{}".format(word))
                continue

        np.save(output_file,text_wordvec)
        return text_wordvec
    else:
        text_wordvec = np.load(output_file,allow_pickle=True)
        return text_wordvec



def data_generator(input_X,input_X_word,label_Y):
    # input_X = np.load(text_path,allow_pickle=True)
    # label_Y = np.load(label_path,allow_pickle=True)
    for index in range(len(input_X)):
        text_x = input_X[index]
        text_x_word = input_X_word[index]
        label = label_Y[index]
        yield (text_x,text_x_word, len(text_x)),label

def input_fn(input_X,input_X_word, label_Y, is_training, args):
        _shapes = (([None],[None],()), [None])
        _types = ((tf.int32,tf.int32, tf.int32), tf.int32)
        _pads = ((0,0,0), 0)
        ds = tf.data.Dataset.from_generator(
            lambda: data_generator(input_X,input_X_word, label_Y),
            output_shapes=_shapes,
            output_types=_types, )
        if is_training:
            # input_X = np.load(data_loader.train_X_path,allow_pickle=True)
            # label_Y = np.load(data_loader.train_Y_path,allow_pickle=True)
            ds = ds.shuffle(args.shuffle_buffer).repeat()

        ds = ds.padded_batch(args.train_batch_size, _shapes, _pads)
        ds = ds.prefetch(args.pre_buffer_size)

        return ds

def data_generator_bert(input_X,label_Y):
    # input_X = np.load(text_path,allow_pickle=True)
    # label_Y = np.load(label_path,allow_pickle=True)
    for index in range(len(input_X)):
        text_x = input_X[index]
        label = label_Y[index]
        yield (text_x, len(text_x)),label

def input_bert_fn(input_X, label_Y, is_training, args):
    _shapes = (([None], ()), [None])
    _types = ((tf.int32, tf.int32), tf.int32)
    _pads = ((0,0), 0)
    ds = tf.data.Dataset.from_generator(
        lambda: data_generator_bert(input_X, label_Y),
        output_shapes=_shapes,
        output_types=_types, )
    if is_training:
        # input_X = np.load(data_loader.train_X_path,allow_pickle=True)
        # label_Y = np.load(data_loader.train_Y_path,allow_pickle=True)
        ds = ds.shuffle(args.shuffle_buffer).repeat()

    ds = ds.padded_batch(args.train_batch_size, _shapes, _pads)
    ds = ds.prefetch(args.pre_buffer_size)

    return ds

def data_generator_bert_mrc(input_Xs,start_Ys,end_Ys,token_type_ids,query_lens):
    for index in range(len(input_Xs)):
        input_x = input_Xs[index]
        start_y = start_Ys[index]
        end_y = end_Ys[index]
        token_type_id = token_type_ids[index]
        query_len = query_lens[index]
        yield (input_x,len(input_x),query_len,token_type_id),(start_y,end_y)

def input_bert_mrc_fn(input_Xs,start_Ys,end_Ys,token_type_ids,query_lens,is_training,args):
    _shapes = (([None], (),(),[None]), ([None],[None]))
    _types = ((tf.int32,tf.int32,tf.int32,tf.int32),(tf.int32,tf.int32))
    _pads = ((0,0,0,0),(0,0))
    ds = tf.data.Dataset.from_generator(
        lambda: data_generator_bert_mrc(input_Xs, start_Ys,end_Ys,token_type_ids,query_lens),
        output_shapes=_shapes,
        output_types=_types, )
    if is_training:
        ds = ds.shuffle(args.shuffle_buffer).repeat()

    ds = ds.padded_batch(args.train_batch_size, _shapes, _pads)
    ds = ds.prefetch(args.pre_buffer_size)

    return ds

def create_dataset_for_torch(features):
    all_input_ids = torch.tensor(features.get("input_ids"),dtype=torch.long)
    all_input_mask = torch.tensor(features.get("input_mask"),dtype=torch.long)
    all_segment_ids = torch.tensor(features.get("segment_ids"),dtype=torch.long)
    all_label_ids = torch.tensor(features.get("label_ids"),dtype=torch.long)
    all_input_lens = torch.tensor(features.get("input_lens"),dtype=torch.long)
    print(type(features.get("word_seq_len_baidu")))
    all_word_seq_lens_baidu = torch.tensor(features.get("word_seq_len_baidu"),dtype=torch.long)
    all_word_seq_lens_thu = torch.tensor(features.get("word_seq_len_thu"),dtype=torch.long)
    all_word_seq_lens_ltp = torch.tensor(features.get("word_seq_len_ltp"),dtype=torch.long)
    all_word_slice_baidu = torch.tensor(features.get("word_slice_baidu"),dtype=torch.long)
    all_word_slice_thu = torch.tensor(features.get("word_slice_thu"),dtype=torch.long)
    all_word_slice_ltp = torch.tensor(features.get("word_slice_ltp"),dtype=torch.long)
    dataset = TensorDataset(all_input_ids,all_word_seq_lens_baidu,all_word_seq_lens_thu,all_word_seq_lens_ltp,all_word_slice_baidu,all_word_slice_thu,all_word_slice_ltp,
    all_segment_ids,all_input_mask,all_label_ids,all_input_lens)
    return dataset