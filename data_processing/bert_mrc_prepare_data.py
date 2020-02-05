import os
import copy
import codecs
import numpy as np
from data_processing.basic_prepare_data import BaseDataPreparing
from data_processing.mrc_query_map import ner_query_map


class bertMRCPrepareData(BaseDataPreparing):
    def __init__(self,vocab_file,slot_file,config,bert_file,max_length,gen_new_data=False,is_inference=False,direct_cut=False):
        self.bert_file = bert_file
        self.max_length = max_length
        self.query_map_dict = self.gen_query_map_dict()
        self.direct = direct_cut
        super(bertMRCPrepareData,self).__init__(vocab_file,slot_file,config,pretrained_embedding_file=None,word_embedding_file=None,load_w2v_embedding=False,load_word_embedding=False,gen_new_data=gen_new_data,is_inference=is_inference)

    def init_final_data_path(self,config,load_word_embedding=False):
        root_path = config.get("data_dir") + "/" + config.get("train_valid_data_dir")
        if not os.path.exists(root_path):
            os.mkdir(root_path)
        self.train_X_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"), config.get("train_data_text_name"))
        self.valid_X_path = os.path.join(config.get("data_dir"),config.get("train_valid_data_dir"), config.get("valid_data_text_name"))
        self.train_start_Y_path = os.path.join(config.get("data_dir"),config.get("train_valid_data_dir"), config.get("train_data_start_tag_name"))
        self.train_end_Y_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"),
                                               config.get("train_data_end_tag_name"))
        self.valid_start_Y_path = os.path.join(config.get("data_dir"),config.get("train_valid_data_dir"), config.get("valid_data_start_tag_name"))
        self.valid_end_Y_path = os.path.join(config.get("data_dir"),config.get("train_valid_data_dir"), config.get("valid_data_end_tag_name"))
        self.test_X_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"), config.get("test_data_text_name"))
        self.train_token_type_ids_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"), config.get("train_data_token_type_ids_name"))
        self.valid_token_type_ids_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"), config.get("valid_data_token_type_ids_name"))
        self.test_token_type_ids_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"), config.get("test_data_token_type_ids_name"))

        self.train_query_len_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"), config.get("train_data_query_len_name"))
        self.valid_query_len_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"), config.get("valid_data_query_len_name"))
        self.test_query_len_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"), config.get("test_data_query_len_name"))
        self.test_query_class_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"), config.get("test_data_query_class"))
        self.src_test_sample_id_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"), config.get("test_data_src_sample_id"))

        # self.test_Y_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"), config.get("test_data_tag_name"))

    def split_one_sentence_based_on_length(self, texts, text_allow_length, labels=None):
        # 通用的截断
        data_list = []
        data_label_list = []
        if len(texts) > text_allow_length:
            left_length = 0
            while left_length+text_allow_length < len(texts):
                cur_cut_index = left_length + text_allow_length
                if labels != None:
                    last_label_tmp = labels[cur_cut_index-1]
                    if last_label_tmp.upper() != "O":
                        while labels[cur_cut_index - 1].upper() != "O":
                            cur_cut_index -= 1
                    data_label_list.append(labels[left_length:cur_cut_index])
                data_list.append(texts[left_length:cur_cut_index])
                left_length = cur_cut_index

            # 别忘了最后还有余下的一小段没处理
            if labels != None:
                data_label_list.append(labels[left_length:])
            data_list.append(texts[left_length:])
        else:
            data_list.append(texts)
            data_label_list.append(labels)
        return data_list,data_label_list


    def split_one_sentence_based_on_entity_direct(self,texts,text_allow_length,labels=None):
        # 对于超过长度的直接截断
        data_list = []
        data_label_list = []
        if len(texts) > text_allow_length:
            pick_texts = texts[0:text_allow_length]
            data_list.append(pick_texts)
            if labels != None:
                data_label_list.append(labels[0:text_allow_length])
                return data_list, data_label_list
        else:
            data_list.append(texts)
            data_label_list.append(labels)
            return data_list, data_label_list

    def token_data_text(self,text):
        word_list = []
        if self.is_inference:
            word_list.extend([w for w in text if w !=" "])
        else:
            word_list.extend(self.tokenizer.tokenize(text))
        return word_list

    def gen_query_map_dict(self):
        slot_query_tag_dict = {}
        for slot_tag in ner_query_map.get("tags"):
            slot_query = ner_query_map.get("natural_query").get(slot_tag)
            slot_query_tokenize = [w for w in slot_query]
            slot_query_tokenize.insert(0, "[CLS]")
            slot_query_tokenize.append("[SEP]")
            slot_query_tag_dict.update({slot_tag:slot_query_tokenize})
        return slot_query_tag_dict

    def find_tag_start_end_index(self,tag,label_list):
        start_index_tag = [0] * len(label_list)
        end_index_tag = [0] * len(label_list)
        start_tag = "B-"+tag
        end_tag = "I-"+tag
        for i in range(len(start_index_tag)):
            if label_list[i].upper() == start_tag:
                # begin
                start_index_tag[i] = 1
            elif label_list[i].upper() == end_tag:
                if i == len(start_index_tag)-1:
                    # last tag
                    end_index_tag[i] = 1
                else:
                    if label_list[i+1].upper() != end_tag:
                        end_index_tag[i] = 1
        return start_index_tag,end_index_tag


    def trans_orig_data_to_training_data(self,datas_file):
        data_X = []
        data_start_Y = []
        data_end_Y = []
        token_type_ids_list = []
        query_len_list = []
        with codecs.open(datas_file,'r','utf-8') as fr:
            tmp_text_split = None
            for index,line in enumerate(fr):
                line = line.strip("\n")
                if index % 2 == 0:
                    tmp_text_split = self.token_data_text(line)
                else:
                    slot_label_list = self.tokenizer.tokenize(line)
                    for slot_tag in self.query_map_dict:
                        slot_query = self.query_map_dict.get(slot_tag)
                        slot_query = [w for w in slot_query]
                        query_len = len(slot_query)
                        text_allow_max_len = self.max_length - query_len
                        if not self.direct:
                            gen_tmp_X_texts,gen_tmp_y_labels = self.split_one_sentence_based_on_length(
                                tmp_text_split,text_allow_max_len,slot_label_list)
                        else:
                            gen_tmp_X_texts, gen_tmp_y_labels = self.split_one_sentence_based_on_entity_direct(
                                tmp_text_split,text_allow_max_len,slot_label_list)
                        for tmp_X,tmp_Y in zip(gen_tmp_X_texts,gen_tmp_y_labels):
                            x_merge = slot_query + tmp_X
                            token_type_ids = [0]*len(slot_query) + [1]*(len(tmp_X))
                            x_merge = self.tokenizer.convert_tokens_to_ids(x_merge)
                            data_X.append(x_merge)
                            start_index_tag,end_index_tag = self.find_tag_start_end_index(slot_tag,tmp_Y)
                            # print(len(x_merge))
                            # print(len(start_index_tag))
                            start_index_tag = [0]*len(slot_query) + start_index_tag
                            # print(len(start_index_tag))
                            end_index_tag = [0]*len(slot_query) + end_index_tag
                            data_start_Y.append(start_index_tag)
                            data_end_Y.append(end_index_tag)
                            token_type_ids_list.append(token_type_ids)
                            query_len_list.append(query_len)
        return data_X,data_start_Y,data_end_Y,token_type_ids_list,query_len_list

    def gen_train_dev_from_orig_data(self,gen_new):
        if gen_new:
            train_data_X,train_data_start_Y,train_data_end_Y,train_token_type_ids_list,train_query_len_list = self.trans_orig_data_to_training_data(self.train_data_file)
            dev_data_X,dev_data_start_Y,dev_data_end_Y,dev_token_type_ids_list,dev_query_len_list = self.trans_orig_data_to_training_data(self.dev_data_file)
            # test_data_X,test_data_start_Y,test_data_end_Y,test_token_type_ids_list,test_query_len_list = self.trans_orig_data_to_training_data(self.test_data_file)
            # dev_data_X = np.concatenate((dev_data_X,test_data_X),axis=0)
            # dev_data_Y = np.concatenate((dev_data_Y,test_data_Y),axis=0)
            self.train_samples_nums = len(train_data_X)
            self.eval_samples_nums = len(dev_data_X)
            np.save(self.train_X_path, train_data_X)
            np.save(self.valid_X_path, dev_data_X)
            np.save(self.train_start_Y_path,train_data_start_Y)
            np.save(self.train_end_Y_path, train_data_end_Y)
            np.save(self.valid_start_Y_path, dev_data_start_Y)
            np.save(self.train_start_Y_path, train_data_start_Y)
            np.save(self.valid_end_Y_path,dev_data_end_Y)
            np.save(self.train_token_type_ids_path, train_token_type_ids_list)
            np.save(self.valid_token_type_ids_path, dev_token_type_ids_list)
            np.save(self.train_query_len_path,train_query_len_list)
            np.save(self.valid_query_len_path,dev_query_len_list)
            # np.save(self.test_X_path,test_data_X)
            # np.save(self.test_token_type_ids_path, test_token_type_ids_list)
            # np.save(self.test_query_len_path, test_query_len_list)
            # np.save(self.test_Y_path,test_data_Y)
        else:
            train_data_X = np.load(self.train_X_path)
            dev_data_X = np.load(self.valid_X_path)
            self.train_samples_nums = len(train_data_X)
            self.eval_samples_nums = len(dev_data_X)

    def trans_test_data(self):
            self.gen_test_data_from_orig_data(self.test_data_file)

    def gen_test_data_from_orig_data(self,datas_file):
        # 相对于训练集来说，测试集构造数据要更复杂一点
        # 1、query要标明，2、因长度问题分割的句子最后要拼起来，因此同一个原样本的要标明 3、最后要根据query对应的实体类别根据start end 关系拼起来
        data_X = []
        token_type_ids_list = []
        query_len_list = []
        query_class_list = []
        src_test_sample_id = []
        with codecs.open(datas_file, 'r', 'utf-8') as fr:
            tmp_text_split = None
            for index, line in enumerate(fr):
                line = line.strip("\n")
                if index % 2 == 0:
                    tmp_text_split = self.token_data_text(line)
                    cur_sample_id = int(index / 2)
                    for slot_tag in self.query_map_dict:
                        slot_query = self.query_map_dict.get(slot_tag)
                        slot_query = [w for w in slot_query]
                        query_len = len(slot_query)
                        text_allow_max_len = self.max_length - query_len
                        gen_tmp_X_texts, _ = self.split_one_sentence_based_on_length(
                            tmp_text_split, text_allow_max_len)
                        for tmp_X in gen_tmp_X_texts:
                            x_merge = slot_query + tmp_X
                            token_type_ids = [0] * len(slot_query) + [1] * (len(tmp_X))
                            x_merge = self.tokenizer.convert_tokens_to_ids(x_merge)
                            data_X.append(x_merge)
                            src_test_sample_id.append(cur_sample_id)
                            query_class_list.append(ner_query_map.get("tags").index(slot_tag))
                            token_type_ids_list.append(token_type_ids)
                            query_len_list.append(query_len)
        np.save(self.test_X_path,data_X)
        np.save(self.test_token_type_ids_path, token_type_ids_list)
        np.save(self.test_query_len_path, query_len_list)
        np.save(self.test_query_class_path,query_class_list)
        np.save(self.src_test_sample_id_path,src_test_sample_id)
