import os
import codecs
import gensim
import numpy as np
import tensorflow as tf

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

def data_generator(input_X,label_Y):
    # input_X = np.load(text_path,allow_pickle=True)
    # label_Y = np.load(label_path,allow_pickle=True)
    for index in range(len(input_X)):
        text_x = input_X[index]
        label = label_Y[index]
        yield (text_x, len(text_x)),label

def input_fn(input_X, label_Y, is_training, args):
        _shapes = (([None], ()), [None])
        _types = ((tf.int32, tf.int32), tf.int32)
        _pads = ((0, 0), 0)
        ds = tf.data.Dataset.from_generator(
            lambda: data_generator(input_X, label_Y),
            output_shapes=_shapes,
            output_types=_types, )
        if is_training:
            # input_X = np.load(data_loader.train_X_path,allow_pickle=True)
            # label_Y = np.load(data_loader.train_Y_path,allow_pickle=True)
            ds = ds.shuffle(args.shuffle_buffer).repeat(args.epochs)

        ds = ds.padded_batch(args.train_batch_size, _shapes, _pads)
        ds = ds.prefetch(args.pre_buffer_size)

        return ds
