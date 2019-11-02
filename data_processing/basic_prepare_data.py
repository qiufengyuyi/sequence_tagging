import os
import codecs
import re
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold,train_test_split
from data_processing.data_utils import gen_char_embedding
from data_processing.tokenize import CustomTokenizer
from configs.base_config import config


class BaseDataPreparing(object):
    def __init__(self,vocab_file,slot_file,pretrained_embedding_file=None,word_embedding_file=None,load_w2v_embedding=True,gen_new_data=False,is_inference=False):
        self.gen_new_data = gen_new_data
        self.train_data_file = os.path.join(config.get("data_dir"),config.get("data_file_name"))
        self.dev_data_file = os.path.join(config.get("data_dir"),config.get("orig_dev"))
        self.test_data_file = os.path.join(config.get("data_dir"), config.get("orig_test"))
        self.train_valid_split_data_path = config.get("train_valid_data_dir")
        self.tokenizer = CustomTokenizer(vocab_file,slot_file)
        self.train_X_path,self.valid_X_path,self.train_Y_path,self.valid_Y_path,self.test_X_path,self.test_Y_path=None,None,None,None,None,None
        self.pos_slot_list = [value for key,value in self.tokenizer.slot2id.items() if key != "O" and key !="PAD"]
        self.slot_label_size = len(self.tokenizer.slot2id)
        if load_w2v_embedding:
            self.word_embedding = gen_char_embedding(pretrained_embedding_file,self.tokenizer.vocab,output_file=word_embedding_file)
        self.init_final_data_path()
        self.train_samples_nums = 0
        self.is_inference = is_inference
        print("preprocessing data....")
        if not is_inference:
            if gen_new_data:
                self.data_preprocessing()
            else:
                self.load_train_dev_from_orig_data()

    def init_final_data_path(self):
        self.train_X_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"), config.get("train_data_text_name"))
        self.valid_X_path = os.path.join(config.get("data_dir"),config.get("train_valid_data_dir"), config.get("valid_data_text_name"))
        self.train_Y_path = os.path.join(config.get("data_dir"),config.get("train_valid_data_dir"), config.get("train_data_tag_name"))
        self.valid_Y_path = os.path.join(config.get("data_dir"),config.get("train_valid_data_dir"), config.get("valid_data_tag_name"))
        self.test_X_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"), config.get("test_data_text_name"))
        self.test_Y_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"), config.get("test_data_tag_name"))

    def tranform_singlg_data_example(self,text):
        if self.is_inference:
            word_list = [w for w in text]
        else:
            word_list = self.tokenizer.tokenize(text)
        word_id_list = self.tokenizer.convert_tokens_to_ids(word_list)
        return word_id_list

    def translate_id_2_slot(self,text,label_list):
        entity_list = []
        text_list = [w for w in text]
        tmp_entity = ""
        tmp_entity_type = ""
        for char,label in zip(text_list,label_list):
            label_string = self.tokenizer.id2slot.get(label)
            if label_string == "O":
                if tmp_entity != "":
                    entity_list.append({tmp_entity:tmp_entity_type})
                tmp_entity = ""
                tmp_entity_type = ""
            elif label_string == "PAD":
                break
            else:
                tmp_entity += char
                tmp_entity_type = re.split("-",label_string)[-1]
        return entity_list


    def trans_orig_data_to_training_data(self,datas_file):
        data_X = []
        data_Y = []
        with codecs.open(datas_file,'r','utf-8') as fr:
            for index,line in enumerate(fr):
                line = line.strip("\n")
                if index % 2 == 0:
                    data_X.append(self.tranform_singlg_data_example(line))
                else:
                    slot_list = self.tokenizer.tokenize(line)
                    slot_list = [slots.upper() for slots in slot_list]
                    slot_id_list = self.tokenizer.convert_slot_to_ids(slot_list)
                    data_Y.append(slot_id_list)
        return data_X,data_Y


    def data_preprocessing(self,random_seed=2):
        """
        数据预处理，包括原始数据转化和训练集验证机的拆分
        :return:
        """
        data_X,data_Y = self.trans_orig_data_to_training_data(self.train_data_file)

        X_train, X_valid, y_train, y_valid = train_test_split(data_X,data_Y,test_size=0.1, random_state=random_seed)
        self.train_samples_nums = len(X_train)
        np.save(self.train_X_path,X_train)
        np.save(self.valid_X_path, X_valid)
        np.save(self.train_Y_path, y_train)
        np.save(self.valid_Y_path, y_valid)

    def load_train_dev_from_orig_data(self):
        train_data_X,train_data_Y = self.trans_orig_data_to_training_data(self.train_data_file)
        dev_data_X,dev_data_Y = self.trans_orig_data_to_training_data(self.dev_data_file)
        test_data_X,test_data_Y = self.trans_orig_data_to_training_data(self.test_data_file)
        dev_data_X = np.concatenate((dev_data_X,test_data_X),axis=0)
        dev_data_Y = np.concatenate((dev_data_Y,test_data_Y),axis=0)
        self.train_samples_nums = len(train_data_X)
        np.save(self.train_X_path, train_data_X)
        np.save(self.valid_X_path, dev_data_X)
        np.save(self.train_Y_path,train_data_Y)
        np.save(self.valid_Y_path,dev_data_Y)

        np.save(self.test_X_path,test_data_X)
        np.save(self.test_Y_path,test_data_Y)

