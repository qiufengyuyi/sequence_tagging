import tensorflow as tf
import re
import os
import numpy as np
import codecs
import copy
from configs.base_config import config
from configs.bert_config import bert_config
from configs.bert_mrc_config import bert_mrc_config
from tensorflow.contrib import predictor
from data_processing.basic_prepare_data import BaseDataPreparing
from data_processing.bert_prepare_data import bertPrepareData
from data_processing.bert_mrc_prepare_data import bertMRCPrepareData
from data_processing.mrc_query_map import ner_query_map
from pathlib import Path
from argparse import ArgumentParser

class fastPredict(object):
    def __init__(self,model_path,config):
        self.model_path = model_path
        self.data_loader = self.init_data_loader(config)
        self.predict_fn = self.load_models()
        self.config = config

    def init_data_loader(self,config):
        vocab_file_path = os.path.join(config.get("embedding_dir"), config.get("vocab_file"))
        pretrained_embedding_path = os.path.join(config.get("embedding_dir"), config.get("embedding_file_name"))
        input_embedding_path = os.path.join(config.get("embedding_dir"), config.get("input_embedding_file"))
        input_word_embedding_path = os.path.join(config.get("embedding_dir"), config.get("input_word_embedding_file"))
        slot_file = os.path.join(config.get("slot_list_root_path"), config.get("slot_file_name"))

        data_loader = BaseDataPreparing(vocab_file_path, slot_file,config,pretrained_embedding_path, input_embedding_path,input_word_embedding_path,
                                        load_w2v_embedding=True,load_word_embedding=True,gen_new_data=False,is_inference=True)
        return data_loader

    def load_models(self):
        subdirs = [x for x in Path(self.model_path).iterdir()
                   if x.is_dir() and 'temp' not in str(x)]
        latest = str(sorted(subdirs)[-1])
        predict_fn = predictor.from_saved_model(latest)
        return predict_fn

    def predict_text(self,text,word_id_text=None,orig_labels=None,raw_test_data=True,using_bert=False,return_raw_result=False):
        if raw_test_data:
            text = re.sub(" ", "", text)
            words = self.data_loader.tranform_singlg_data_example(text)
            if not using_bert:
                words_char_ids = self.data_loader.gen_one_sample_words_on_chars(text)
        else:
            words = text
            if not using_bert:
                words_char_ids = word_id_text
        nwords = len(words)
        if not using_bert:
            predictions = self.predict_fn({'words': [words],'words_seq':[words_char_ids], 'text_length': [nwords]})
        else:
            predictions = self.predict_fn({'words': [words], 'text_length': [nwords]})
        label_list = predictions["output"][0]
        # print("")
        if using_bert:
            label_list = [label for label in label_list]

            if len(label_list) < len(orig_labels):
                # bert 截断了

                pad_list = [self.data_loader.tokenizer.slot2id["O"]]*(len(orig_labels)-len(label_list))
                label_list.extend(pad_list)
                # print(label_list)
        if return_raw_result:
            # if len(orig_labels)!=len(label_list):
            #     print(len(orig_labels))
            #     print(len(label_list))
            #     print(text)
            #     print(label_list)

            return label_list
        else:
            entity_list = self.data_loader.translate_id_2_slot(text,label_list)
            return entity_list


class fastPredictionBert(fastPredict):
    def __init__(self,model_path,config,label_less):
        # self.config = config
        self.orig_test_file = os.path.join(config.get("data_dir"),config.get("orig_test"))
        self.label_less = label_less
        super(fastPredictionBert,self).__init__(model_path,config)

    def get_orig_test_label(self):
        label_lists = []
        with codecs.open(self.orig_test_file,'r','utf-8') as fr:
            for i,line in enumerate(fr):
                if i % 2 == 1:
                    line = line.strip("\n")
                    # line = re.sub(" ", "", line)
                    labels = self.data_loader.tokenizer.tokenize(line)
                    slot_list = [slots.upper() for slots in labels]
                    # if not self.label_less:
                    #     slot_list.insert(0,"[CLS]")
                    #     slot_list.append("[SEP]")

                    slot_list.insert(0, "O")
                    slot_list.append("O")
                    slot_id_list = self.data_loader.tokenizer.convert_slot_to_ids(slot_list)
                    # labels = [label for label in line]
                    label_lists.append(slot_id_list)
        return label_lists

    def init_data_loader(self,config):
        vocab_file_path = os.path.join(config.get("bert_pretrained_model_path"), config.get("vocab_file"))
        slot_file = os.path.join(config.get("slot_list_root_path"), config.get("bert_slot_complete_file_name"))
        bert_config_file = os.path.join(config.get("bert_pretrained_model_path"), config.get("bert_config_path"))
        data_loader = bertPrepareData(vocab_file_path,slot_file,config,bert_config_file,384,gen_new_data=False,is_inference=True,label_less=self.label_less)
        return data_loader

class fastPredictBertMrc(fastPredict):
    def __init__(self,model_path,config):
        # self.config = config
        self.orig_test_file = os.path.join(config.get("data_dir"),config.get("orig_test"))
        super(fastPredictBertMrc,self).__init__(model_path,config)

    def init_data_loader(self,config):
        vocab_file_path = os.path.join(config.get("bert_pretrained_model_path"), config.get("vocab_file"))
        slot_file = os.path.join(config.get("slot_list_root_path"), config.get("bert_slot_complete_file_name"))
        bert_config_file = os.path.join(config.get("bert_pretrained_model_path"), config.get("bert_config_path"))
        #vocab_file,slot_file,config,bert_file,max_length,gen_new_data=False,is_inference=False
        data_loader = bertMRCPrepareData(vocab_file_path,slot_file,config,bert_config_file,512,gen_new_data=True,is_inference=True)
        return data_loader

    def predict_mrc(self,text,query_len,token_type_ids):
        text_length = len(text)
        # features['words'],features['text_length'],features['query_length'],features['token_type_ids']
        predictions = self.predict_fn({'words': [text], 'text_length': [text_length],'query_length':[query_len],'token_type_ids':[token_type_ids]})
        start_ids,end_ids = predictions.get("start_ids"),predictions.get("end_ids")
        return start_ids[0],end_ids[0]

    def extract_entity_from_start_end_ids(self,orig_text,start_ids,end_ids):
        # 根据开始，结尾标识，找到对应的实体
        entity_list = []
        for i,start_id in enumerate(start_ids):
            if start_id == 0:
                continue
            j = i+1
            find_end_tag = False
            while j < len(end_ids):
                # 若在遇到end=1之前遇到了新的start=1,则停止该实体的搜索
                if start_ids[j] == 1:
                    break
                if end_ids[j] == 1:
                    entity_list.append("".join(orig_text[i:j+1]))
                    find_end_tag = True
                    break
                else:
                    j+=1
            if not find_end_tag:
                # 实体就一个单字
                entity_list.append("".join(orig_text[i:i+1]))
        return entity_list

    def predict_entitys_for_all_sample(self,text_data_Xs,query_lens,token_type_ids_list,query_class_list,src_sample_ids_list,orig_text_list):
        result_list = [] # 存储的是每个样本每个实体类别对应的实体列表，有可能是空的
        cur_sample_id_buffer = 0
        start_ids_buffer = []
        end_ids_buffer = []
        query_class_buffer = ""
        for i in range(len(text_data_Xs)):
            cur_text = text_data_Xs[i]
            cur_query_len = query_lens[i]
            cur_token_type_ids = token_type_ids_list[i]
            cur_query_class = query_class_list[i]
            cur_src_sample_id = src_sample_ids_list[i]
            start_ids,end_ids = self.predict_mrc(cur_text,cur_query_len,cur_token_type_ids)
            # 去掉query
            # print(type(start_ids))
            true_start_ids = start_ids[cur_query_len:].tolist()
            true_end_ids = end_ids[cur_query_len:].tolist()
            cur_query_class_str = ner_query_map.get("tags")[cur_query_class]
            if query_class_buffer == "" or len(start_ids_buffer)==0:
                # 首个样本，都添加到buffer中
                query_class_buffer = cur_query_class_str
                start_ids_buffer.extend(true_start_ids)
                end_ids_buffer.extend(true_end_ids)
                cur_sample_id_buffer = cur_src_sample_id
            elif cur_src_sample_id == cur_sample_id_buffer and cur_query_class_str == query_class_buffer:
                # 同一个样本,同一个query,要合并
                start_ids_buffer.extend(true_start_ids)
                end_ids_buffer.extend(true_end_ids)
            elif cur_src_sample_id == cur_sample_id_buffer:
                # 遇到不同query 类型，先处理上一个query类型的样本实体识别
                cur_orig_text = orig_text_list[cur_sample_id_buffer]

                extracted_entity_list = self.extract_entity_from_start_end_ids(cur_orig_text,start_ids_buffer,end_ids_buffer)
                # print(result_list)
                # print(cur_src_sample_id)
                # print(cur_orig_text)
                if len(result_list) == 0:
                    # 初始情况
                    # buffer 的query class 更新
                    result_list.append({query_class_buffer:extracted_entity_list})
                else:
                    if cur_sample_id_buffer >= len(result_list):
                        result_list.append({query_class_buffer: extracted_entity_list})
                    else:
                        result_list[cur_sample_id_buffer].update({query_class_buffer:extracted_entity_list})
                # 更新query_class_buffer
                query_class_buffer = cur_query_class_str
                # 更新start_ids_buffer，end_ids_buffer
                start_ids_buffer = true_start_ids
                end_ids_buffer = true_end_ids
            else:
                # 本轮为新的样本
                cur_orig_text = orig_text_list[cur_sample_id_buffer]
                extracted_entity_list = self.extract_entity_from_start_end_ids(cur_orig_text, start_ids_buffer,
                                                                               end_ids_buffer)
                # if cur_src_sample_id == 2:
                #     print(extracted_entity_list)
                # 更新上一个id的样本实体抽取
                # print(cur_sample_id_buffer)
                # print(result_list)
                if cur_sample_id_buffer >= len(result_list):
                    result_list.append({query_class_buffer: extracted_entity_list})
                else:
                    result_list[cur_sample_id_buffer].update({query_class_buffer: extracted_entity_list})
                query_class_buffer = cur_query_class_str
                start_ids_buffer = true_start_ids
                end_ids_buffer = true_end_ids
                cur_sample_id_buffer = cur_src_sample_id
        # deal with last sample
        cur_orig_text = orig_text_list[cur_sample_id_buffer]
        extracted_entity_list = self.extract_entity_from_start_end_ids(cur_orig_text, start_ids_buffer,
                                                                       end_ids_buffer)
        if cur_sample_id_buffer >= len(result_list):
            result_list.append({query_class_buffer: extracted_entity_list})
        else:
            result_list[cur_sample_id_buffer].update({query_class_buffer: extracted_entity_list})
        return result_list


    def gen_micro_level_entity_span(self,type2entity_list):
        # 将每个样本的所有实体集中起来，先来计算micro level的f1
        entity_list_final = []
        for type2entity in type2entity_list:
            cur_tmp_list = []
            for slot_type,entity_list in type2entity.items():
                cur_tmp_list.extend(entity_list)
            entity_list_final.append(list(set(cur_tmp_list)))
        return entity_list_final

def gen_entity_from_label_id_list(text_lists,label_id_list,id2slot_dict,orig_test=False):
    """
    B-LOC
    B-PER
    B-ORG
    I-LOC
    I-ORG
    I-PER
    :param label_id_list:
    :param id2slot_dict:
    :return:
    """
    entity_list = []
    # 存index
    buffer_list = []
    for i,label_ids in enumerate(label_id_list):
        cur_entity_list = []
        if not orig_test:
            label_list = [id2slot_dict.get(label_ele) for label_ele in label_ids]
        else:
            label_list = label_ids
        text_list = text_lists[i]
        # label_list
        # print(label_list)
        for j,label in enumerate(label_list):
            if not label.__contains__("-"):
                if len(buffer_list)==0:
                    continue
                else:
                    # print(buffer_list)
                    # print(text_list)
                    buffer_char_list = [text_list[index] for index in buffer_list]
                    buffer_word = "".join(buffer_char_list)
                    cur_entity_list.append(buffer_word)
                    buffer_list.clear()
            else:
                if len(buffer_list) == 0:
                    if label.startswith("B"):
                        #必须以B开头，否则说明有问题，不能加入
                        buffer_list.append(j)
                else:
                    buffer_last_index = buffer_list[-1]
                    buffer_last_label = label_list[buffer_last_index]
                    split_label = buffer_last_label.split("-")
                    buffer_last_label_prefix,buffer_last_label_type = split_label[0],split_label[1]
                    cur_label_split = label.split("-")
                    cur_label_prefix,cur_label_type = cur_label_split[0],cur_label_split[1]
                    # B+B
                    if buffer_last_label_prefix=="B" and cur_label_prefix=="B":
                        cur_entity_list.append(text_list[buffer_list[-1]])
                        buffer_list.clear()
                        buffer_list.append(j)
                    elif buffer_last_label_prefix=="I" and cur_label_prefix=="B":
                        buffer_char_list = [text_list[index] for index in buffer_list]
                        buffer_word = "".join(buffer_char_list)
                        cur_entity_list.append(buffer_word)
                        buffer_list.clear()
                        buffer_list.append(j)
                    elif buffer_last_label_prefix=="B" and cur_label_prefix=="I":
                        # analyze type
                        if buffer_last_label_type == cur_label_type:
                            buffer_list.append(j)
                        else:
                            cur_entity_list.append(text_list[buffer_list[-1]])
                            buffer_list.clear()
                            # 这种情况出现在预测有问题，即一个I的label不应当作为一个实体的起始。
                            #buffer_list.append(j)
                    else:
                        # I + I
                        # analyze type
                        if buffer_last_label_type == cur_label_type:
                            buffer_list.append(j)
                        else:
                            cur_entity_list.append(text_list[buffer_list[-1]])
                            buffer_list.clear()
                            buffer_list.append(j)
        if buffer_list:
            buffer_char_list = [text_list[index] for index in buffer_list]
            buffer_word = "".join(buffer_char_list)
            cur_entity_list.append(buffer_word)
            buffer_list.clear()
        entity_list.append(cur_entity_list)
    return entity_list

def gen_orig_test_text_label(has_cls=True):
    orig_text = []
    orig_label = []
    with codecs.open("data/orig_data_test.txt",'r','utf-8') as fr:
        for i, line in enumerate(fr):
            if i % 2 == 0:
                text = line.strip("\n")
                text_split = text.split(" ")
                if has_cls:
                    print(text_split)
                    print(has_cls)
                    text_split.insert(0,"[CLS]")
                    text_split.append("[SEP]")
                orig_text.append(text_split)
            else:
                text = line.strip("\n")
                text_split = text.split(" ")
                if has_cls:
                    text_split.insert(0,"O")
                    text_split.append("O")
                orig_label.append(text_split)
    return orig_text,orig_label

def cal_mertric_from_two_list(prediction_list,true_list):
    tp, fp, fn = 0, 0, 0
    for pred_entity, true_entity in zip(prediction_list, true_list):
        pred_entity_set = set(pred_entity)
        true_entity_set = set(true_entity)
        tp += len(true_entity_set & pred_entity_set)
        fp += len(pred_entity_set - true_entity_set)
        fn += len(true_entity_set - pred_entity_set)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec)
    print("span_level pre micro_avg:{}".format(prec))
    print("span_level rec micro_avg:{}".format(rec))
    print("span_level f1 micro_avg:{}".format(f1))

def cal_span_level_micro_average(args):
    if not os.path.exists(args.prediction_result_path):
        if args.model_type == "bert":
            gen_prediction_output(True,args)
        else:
            gen_prediction_output(False,args)
    prediction_array = np.load(args.prediction_result_path)
    orig_texts, orig_labels = gen_orig_test_text_label(args.has_cls)
    if args.model_type == "bert":

        fp = fastPredictionBert(bert_config.get(args.model_pb_dir), bert_config, args.label_less)
        id2slot_dict = fp.data_loader.tokenizer.id2slot
        true_entity_list = gen_entity_from_label_id_list(orig_texts,orig_labels,id2slot_dict,orig_test=True)
        # print(len(prediction_array))
        # print(len(orig_texts))
        prediction_entity_list = gen_entity_from_label_id_list(orig_texts,prediction_array,id2slot_dict,orig_test=False)
        print(len(true_entity_list))
        print(len(prediction_entity_list))
        cal_mertric_from_two_list(prediction_entity_list,true_entity_list)
    else:
        # fp = fastPredict(config.get(args.model_pb_dir),config)
        # O
        # B - LOC
        # B - PER
        # B - ORG
        # I - LOC
        # I - ORG
        # I - PER
        id2slot_dict = {7:"O",1:"B-LOC",2:"B-PER",3:"B-ORG",4:"I-LOC",5:"I-ORG",6:"I-PER"}
        true_entity_list = gen_entity_from_label_id_list(orig_texts, orig_labels, id2slot_dict, orig_test=True)
        prediction_entity_list = gen_entity_from_label_id_list(orig_texts,prediction_array,id2slot_dict,orig_test=False)
        cal_mertric_from_two_list(prediction_entity_list, true_entity_list)

def cal_span_level_micro_average_for_bertmrc(args):
    fp = fastPredictBertMrc(bert_mrc_config.get(args.model_pb_dir),bert_mrc_config)
    orig_test_text_list,orig_labels = gen_orig_test_text_label(False)
    test_input_ids = np.load(fp.data_loader.test_X_path,allow_pickle=True)
    test_query_lens = np.load(fp.data_loader.test_query_len_path,allow_pickle=True)
    test_token_type_ids = np.load(fp.data_loader.test_token_type_ids_path,allow_pickle=True)
    test_src_sample_id_list = np.load(fp.data_loader.src_test_sample_id_path,allow_pickle=True)
    test_query_class_list = np.load(fp.data_loader.test_query_class_path,allow_pickle=True)
    type2entity_list = fp.predict_entitys_for_all_sample(test_input_ids,test_query_lens,test_token_type_ids,test_query_class_list,
                                      test_src_sample_id_list,orig_test_text_list)
    # print(type2entity_list)

    prediction_list = fp.gen_micro_level_entity_span(type2entity_list)
    print(len(prediction_list))
    # print(prediction_list[0:10])
    id2slot_dict = fp.data_loader.tokenizer.id2slot
    true_entity_list = gen_entity_from_label_id_list(orig_test_text_list, orig_labels, id2slot_dict, orig_test=True)
    print(len(true_entity_list))
    # print(true_entity_list[0:10])
    cal_mertric_from_two_list(prediction_list, true_entity_list)

def gen_prediction_output(using_bert,args):
    if using_bert:
        fp = fastPredictionBert(bert_config.get(args.model_pb_dir), bert_config, args.label_less)
    else:
        fp = fastPredict(bert_config.get(args.model_pb_dir), bert_config, args.label_less)
        test_data_word_X = np.load(fp.data_loader.test_word_path)
    orig_label_list = fp.get_orig_test_label()
    orig_label_array = np.array(orig_label_list)
    np.save("bert_true_label.npy", orig_label_array)
    true_labels = np.load("bert_true_label.npy")
    test_data_X = np.load(fp.data_loader.test_X_path)
    # test_data_word_X = np.load(fp.data_loader.test_word_path)
    # print(test_data_X)
    prediction_list = []
    for i, text in enumerate(test_data_X):
        # print(text.shape)
        # test_data_word = test_data_word_X[i]
        if using_bert:
            prediction_list.append(
                fp.predict_text(text, None, orig_labels=true_labels[i], raw_test_data=False, using_bert=True,
                                return_raw_result=True))
        else:
            test_data_word = test_data_word_X[i]
            prediction_list.append(fp.predict_text(text,test_data_word,None,False,False,True))
    prediction_array = np.array(prediction_list)
    np.save(args.prediction_result_path, prediction_array)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--prediction_result_path", default='prediction_result_baseline.npy', type=str)
    parser.add_argument("--model_pb_dir", default='base_pb_model_dir', type=str)
    parser.add_argument("--model_type", default='bert', type=str)
    parser.add_argument("--label_less", action='store_true',default=False)
    parser.add_argument("--has_cls", action='store_true', default=False)
    # parser.add_argument("--model_pb_dir", default='base_pb_model_dir', type=str)
    args = parser.parse_args()
    # print(args.label_less)
    if args.model_type == "bert":
        cal_span_level_micro_average(args)
    elif args.model_type == "bert_mrc":
        cal_span_level_micro_average_for_bertmrc(args)
    else:
        cal_span_level_micro_average(args)