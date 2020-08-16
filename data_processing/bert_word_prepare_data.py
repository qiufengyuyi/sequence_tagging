import codecs
import os
import json
import pickle
import numpy as np
from data_processing.tokenize import WordObj
from data_processing.data_utils import gen_char_embedding
from data_processing.tokenize import CustomTokenizer
from data_processing.lstmcrf_prepare_data import BaseDataPreparing


class bertWordPrepareData(BaseDataPreparing):
    def __init__(self, vocab_file, slot_file, config, bert_file, max_length, pretrained_embedding_file=None,
                 word_seq_embedding_file=None, gen_new_data=False, is_inference=False, label_less=True):
        self.bert_file = bert_file
        self.max_length = max_length
        self.label_less = label_less
        self.train_data_seg_json = self.load_seg_result_json(config.get("seg_train_data_json"))
        self.dev_data_seg_json = self.load_seg_result_json(config.get("seg_dev_data_json"))
        self.test_data_seg_json = self.load_seg_result_json(config.get("seg_test_data_json"))
        self.train_data_file = os.path.join(config.get("data_dir"), config.get("data_file_name"))
        self.dev_data_file = os.path.join(config.get("data_dir"), config.get("orig_dev"))
        self.test_data_file = os.path.join(config.get("data_dir"), config.get("orig_test"))
        self.train_valid_split_data_path = config.get("train_valid_data_dir")
        self.init_final_data_path(config)
        self.tokenizer = CustomTokenizer(vocab_file, slot_file)
        # if os.path.exists(self.token_obj):
        #     with open(self.token_obj,'rb') as fr:
        #         self.word_statistic = pickle.load(fr)
        # else:
        #     self.word_statistic = WordObj()

        #     self.word_statistic.gen_vocab(self.train_data_seg_json)
        #     self.word_statistic.gen_vocab(self.dev_data_seg_json)
        #     self.word_statistic.gen_vocab(self.test_data_seg_json)

        if not is_inference:
            self.load_word_char_from_orig_data(gen_new_data)
        # self.word_seq_embedding = gen_char_embedding(pretrained_embedding_file,self.word_statistic.vocab,output_file=word_seq_embedding_file)

        super(bertWordPrepareData, self).__init__(vocab_file, slot_file, config, pretrained_embedding_file=None,
                                                  word_embedding_file=None, load_w2v_embedding=False,
                                                  load_word_embedding=False, gen_new_data=gen_new_data,
                                                  is_inference=is_inference)
        if not gen_new_data:
            train_data_X = np.load(self.train_X_path, allow_pickle=True)
            dev_data_X = np.load(self.valid_X_path, allow_pickle=True)
            self.train_samples_nums = len(train_data_X)
            self.eval_samples_nums = len(dev_data_X)

    def load_seg_result_json(self, data_json):
        with codecs.open(data_json, 'r', 'utf-8') as fr:
            data_json = json.load(fr)
        return data_json

    def init_final_data_path(self, config):
        if not os.path.exists(config.get("data_dir") + "/" + config.get("train_valid_data_dir")):
            os.makedirs(config.get("data_dir") + "/" + config.get("train_valid_data_dir"))
        self.train_X_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"),
                                         config.get("train_data_text_name"))
        self.valid_X_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"),
                                         config.get("valid_data_text_name"))
        self.train_Y_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"),
                                         config.get("train_data_tag_name"))
        self.valid_Y_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"),
                                         config.get("valid_data_tag_name"))
        self.test_X_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"),
                                        config.get("test_data_text_name"))
        self.test_Y_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"),
                                        config.get("test_data_tag_name"))
        # self.word_statistic = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"), "bert_word_tokenizer.pkl")
        self.train_word_seq_length_path_baidu = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"),
                                                             "train_word_seq_len_baidu.npy")
        self.train_word_slice_path_baidu = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"),
                                                        "train_word_slice_baidu.npy")
        self.dev_word_seq_length_path_baidu = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"),
                                                           "dev_word_seq_len_baidu.npy")
        self.dev_word_slice_path_baidu = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"),
                                                      "dev_word_slice_baidu.npy")
        self.test_word_seq_length_path_baidu = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"),
                                                            "test_word_seq_len_baidu.npy")
        self.test_word_slice_path_baidu = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"),
                                                       "test_word_slice_baidu.npy")

        self.train_word_seq_length_path_thu = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"),
                                                           "train_word_seq_len_thu.npy")
        self.train_word_slice_path_thu = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"),
                                                      "train_word_slice_thu.npy")
        self.dev_word_seq_length_path_thu = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"),
                                                         "dev_word_seq_len_thu.npy")
        self.dev_word_slice_path_thu = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"),
                                                    "dev_word_slice_thu.npy")
        self.test_word_seq_length_path_thu = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"),
                                                          "test_word_seq_len_thu.npy")
        self.test_word_slice_path_thu = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"),
                                                     "test_word_slice_thu.npy")

        self.train_word_seq_length_path_ltp = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"),
                                                           "train_word_seq_len_ltp.npy")
        self.train_word_slice_path_ltp = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"),
                                                      "train_word_slice_ltp.npy")
        self.dev_word_seq_length_path_ltp = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"),
                                                         "dev_word_seq_len_ltp.npy")
        self.dev_word_slice_path_ltp = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"),
                                                    "dev_word_slice_ltp.npy")
        self.test_word_seq_length_path_ltp = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"),
                                                          "test_word_seq_len_ltp.npy")
        self.test_word_slice_path_ltp = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"),
                                                     "test_word_slice_ltp.npy")

        self.train_input_lens_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"),
                                                  "train_input_lens.npy")
        self.valid_input_lens_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"),
                                                  "valid_input_lens.npy")
        self.test_input_lens_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"),
                                                 "test_input_lens.npy")

        self.train_input_masks_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"),
                                                   "train_input_masks.npy")
        self.valid_input_masks_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"),
                                                   "valid_input_masks.npy")
        self.test_input_masks_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"),
                                                  "test_input_masks.npy")

        self.train_input_segment_ids_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"),
                                                         "train_input_segment_ids.npy")
        self.valid_input_segment_ids_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"),
                                                         "valid_input_segment_ids.npy")
        self.test_input_segment_ids_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"),
                                                        "test_input_segment_ids.npy")

        self.train_Y_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"),
                                         config.get("train_data_tag_name"))
        self.valid_Y_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"),
                                         config.get("valid_data_tag_name"))
        self.test_X_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"),
                                        config.get("test_data_text_name"))
        self.test_Y_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"),
                                        config.get("test_data_tag_name"))

        # self.valid_word_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"), config.get("valid_data_text_word_name"))
        # self.test_word_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"), config.get("test_data_text_word_name"))
        # self.train_word_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"), config.get("train_data_text_word_name"))
        # self.valid_word_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"), config.get("valid_data_text_word_name"))
        # self.test_word_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"), config.get("test_data_text_word_name"))

    def tranform_singlg_data_example(self, text, white_space_sep=False):
        # print(text)
        word_list = []
        # if not self.label_less:
        #     word_list.append("[CLS]")
        word_list.append("[CLS]")

        if not white_space_sep:
            word_list.extend([w for w in text if w != " "])
        else:
            word_list.extend(self.tokenizer.tokenize(text))
        if len(word_list) >= self.max_length:
            word_list = word_list[0:self.max_length - 1]
        # if not self.label_less:
        #     word_list.append("[SEP]")
        word_list.append("[SEP]")
        word_id_list = self.tokenizer.convert_tokens_to_ids(word_list)
        seq_length = len(word_id_list)
        segment_ids = [0] * seq_length
        input_mask = [1] * seq_length
        # while len(word_id_list) < self.max_length:
        #      word_id_list.append(0)
        #      input_mask.append(0)
        #      segment_ids.append(0)
        # print(len(word_id_list))
        return word_id_list, input_mask, segment_ids, seq_length

    def trans_orig_data_to_training_data(self, datas_file):
        data_X = []
        data_Y = []
        data_seq_masks = []
        data_seq_lens = []
        data_segment_ids = []
        with codecs.open(datas_file, 'r', 'utf-8') as fr:
            for index, line in enumerate(fr):
                line = line.strip("\n")
                if index % 2 == 0:
                    word_id_list, input_mask, segment_ids, seq_length = self.tranform_singlg_data_example(line)
                    data_X.append(word_id_list)
                    data_seq_masks.append(input_mask)
                    data_seq_lens.append(seq_length)
                    data_segment_ids.append(segment_ids)
                else:
                    # slot_list = ["[CLS]"]
                    slot_list = []
                    if not self.label_less:
                        slot_list.append("[CLS]")

                    # slot_list.append("O")

                    slot_list.extend(self.tokenizer.tokenize(line))
                    if len(slot_list) >= self.max_length:
                        slot_list = slot_list[0:self.max_length - 1]

                    if not self.label_less:
                        slot_list.append("[SEP]")

                    # slot_list.append("O")

                    slot_list = [slots.upper() for slots in slot_list]
                    slot_id_list = self.tokenizer.convert_slot_to_ids(slot_list)
                    # while len(slot_id_list) < self.max_length:
                    #     slot_id_list.append(0)
                    data_Y.append(slot_id_list)
        return data_X, data_Y, data_seq_masks, data_seq_lens, data_segment_ids

    # def seg_word_for_data(self,data_json,lac_type="baidu"):
    #     for data_ele in data_json:
    #         seg_result = data_ele.get(lac_type)
    #         seg_result_id_list = [self.word_statistic.vocab.get(ele_word) for ele_word in seg_result]
    #     # [CLS]
    #     word_ids_seq_char_list = [1]
    #     for word,word_ids in zip(seg_result,seg_result_id_list):
    #         word_len = len(word)
    #         word_ids_seq_char_list.extend([word_ids]*word_len)
    #     # print(word_ids_seq_char_list)
    #     # assert(len(word_ids_seq_char_list)==len(text.split(" ")))
    #     #[SEP]
    #     word_ids_seq_char_list.append(2)
    #     return word_ids_seq_char_list

    def load_word_char_from_orig_data(self, gen_new):
        if gen_new:
            # train_data_X = np.load(self.train_X_path)
            # dev_data_X = np.load(self.valid_X_path)
            train_data_X, train_data_Y, train_data_input_masks, train_data_input_lens, train_data_segment_ids = self.trans_orig_data_to_training_data(
                self.train_data_file)
            # train_data_X_word = self.seg_word_for_data(self.train_data_seg_json)
            train_word_length_list_baidu, train_word_slice_list_baidu, train_word_length_list_thu, train_word_slice_list_thu, train_word_length_list_ltp, train_word_slice_list_ltp = self.torch_create_seq_length_and_slice(
                self.train_data_seg_json)
            # train_word_cws_labels_list = self.torch_get_cws_label(self.train_data_seg_json,train_data_input_lens)
            dev_data_X, dev_data_Y, dev_data_input_masks, dev_data_input_lens, dev_data_segment_ids = self.trans_orig_data_to_training_data(
                self.dev_data_file)
            # dev_data_X_word = self.seg_word_for_data(self.dev_data_seg_json)
            dev_word_length_list_baidu, dev_word_slice_list_baidu, dev_word_length_list_thu, dev_word_slice_list_thu, dev_word_length_list_ltp, dev_word_slice_list_ltp = self.torch_create_seq_length_and_slice(
                self.dev_data_seg_json)

            test_data_X, test_data_Y, test_data_input_masks, test_data_input_lens, test_data_segment_ids = self.trans_orig_data_to_training_data(
                self.test_data_file)
            # test_data_X_word = self.seg_word_for_data(self.test_data_seg_json)
            test_word_length_list_baidu, test_word_slice_list_baidu, test_word_length_list_thu, test_word_slice_list_thu, test_word_length_list_ltp, test_word_slice_list_ltp = self.torch_create_seq_length_and_slice(
                self.test_data_seg_json)

            # dev_data_X = np.concatenate((dev_data_X, test_data_X), axis=0)
            # dev_data_Y = np.concatenate((dev_data_Y, test_data_Y), axis=0)
            # dev_data_word = np.concatenate((dev_data_X_word,test_data_X_word),axis=0)
            self.train_samples_nums = len(train_data_X)
            self.eval_samples_nums = len(dev_data_X)
            np.save(self.train_X_path, train_data_X)
            np.save(self.valid_X_path, dev_data_X)
            np.save(self.train_Y_path, train_data_Y)
            np.save(self.valid_Y_path, dev_data_Y)

            np.save(self.test_X_path, test_data_X)
            np.save(self.test_Y_path, test_data_Y)
            # np.save(self.train_word_path,train_data_X_word)
            # np.save(self.valid_word_path,dev_data_X_word)
            # np.save(self.test_word_path,test_data_X_word)
            np.save(self.train_word_seq_length_path_baidu, train_word_length_list_baidu)
            np.save(self.train_word_slice_path_baidu, train_word_slice_list_baidu)
            np.save(self.train_word_seq_length_path_thu, train_word_length_list_thu)
            np.save(self.train_word_slice_path_thu, train_word_slice_list_thu)
            np.save(self.train_word_seq_length_path_ltp, train_word_length_list_ltp)
            np.save(self.train_word_slice_path_ltp, train_word_slice_list_ltp)

            np.save(self.dev_word_seq_length_path_baidu, dev_word_length_list_baidu)
            np.save(self.dev_word_slice_path_baidu, dev_word_slice_list_baidu)
            np.save(self.dev_word_seq_length_path_thu, dev_word_length_list_thu)
            np.save(self.dev_word_slice_path_thu, dev_word_slice_list_thu)
            np.save(self.dev_word_seq_length_path_ltp, dev_word_length_list_ltp)
            np.save(self.dev_word_slice_path_ltp, dev_word_slice_list_ltp)

            np.save(self.test_word_seq_length_path_baidu, test_word_length_list_baidu)
            np.save(self.test_word_slice_path_baidu, test_word_slice_list_baidu)
            np.save(self.test_word_seq_length_path_thu, test_word_length_list_thu)
            np.save(self.test_word_slice_path_thu, test_word_slice_list_thu)
            np.save(self.test_word_seq_length_path_ltp, test_word_length_list_ltp)
            np.save(self.test_word_slice_path_ltp, test_word_slice_list_ltp)

            np.save(self.train_input_lens_path, train_data_input_lens)
            np.save(self.train_input_masks_path, train_data_input_masks)
            np.save(self.train_input_segment_ids_path, train_data_segment_ids)
            np.save(self.valid_input_lens_path, dev_data_input_lens)
            np.save(self.valid_input_masks_path, dev_data_input_masks)
            np.save(self.valid_input_segment_ids_path, dev_data_segment_ids)
            np.save(self.test_input_lens_path, test_data_input_lens)
            np.save(self.test_input_masks_path, test_data_input_masks)
            np.save(self.test_input_segment_ids_path, test_data_segment_ids)


        else:
            train_data_X = np.load(self.train_X_path, allow_pickle=True)
            dev_data_X = np.load(self.valid_X_path, allow_pickle=True)
            self.train_samples_nums = len(train_data_X)
            self.eval_samples_nums = len(dev_data_X)

    # def torch_get_cws_label(self,data_json,text_len_list):
    #     #BMES
    #     word_cws_label_ids_list = []
    #     for index,data_ele in enumerate(data_json):

    #         text_len = text_len_list[index]
    #         cur_cws_label_ids = [[0]*5]*text_len
    #         # SEP CLS
    #         cur_cws_label_ids[0] = [0,0,0,0,1]
    #         cur_cws_label_ids[-1] = [0,0,0,0,1]
    #         for lac_type in data_ele:
    #             lac_result = data_ele.get(lac_type)
    #             cur_index = 1
    #             cur_len = 1
    #             for word in lac_result:
    #                 if cur_len+len(word)> text_len - 1:
    #                     if len(word) == 1:
    #                         break
    #                     else:
    #                         word_truncate = word[0:text_len-1-cur_len]
    #                         for ind,cha in enumerate(word_truncate):
    #                             if ind == 0:
    #                                 cur_cws_label_ids[cur_index][1] = 1
    #                             else:
    #                                 cur_cws_label_ids[cur_index][2] = 1
    #                             cur_index += len(word_truncate)
    #                 else:
    #                     if len(word) == 1:
    #                         cur_cws_label_ids[cur_index][4] = 1
    #                         cur_index += 1
    #                         cur_len += 1
    #                     else:
    #                         for ind,cha in enumerate(word):
    #                             if ind == 0:
    #                                 cur_cws_label_ids[cur_index][1] = 1
    #                             elif ind < len(word):
    #                                 cur_cws_label_ids[cur_index][2] = 1
    #                             else:
    #                                 cur_cws_label_ids[cur_index][3] = 1
    #                             cur_index += len(word)
    #         word_cws_label_ids_list.append(cur_cws_label_ids)
    #     return word_cws_label_ids_list

    def torch_create_seq_length_and_slice(self, data_json):
        word_length_list_baidu = []
        word_slice_list_baidu = []
        word_length_list_thu = []
        word_slice_list_thu = []
        word_length_list_ltp = []
        word_slice_list_ltp = []

        def one_type_lac(lac_type):
            seg_result_baidu = data_ele.get(lac_type)
            # seg_result_baidu.insert("[CLS]",0)
            # seg_result_baidu.append("[SEP]")
            cur_word_lenghts = [1]
            cur_word_slices = [0]
            cur_index = 1
            cur_len = 1
            for word in seg_result_baidu:
                word_len = len(word)
                if cur_len + word_len >= self.max_length:
                    cur_word_lenghts.append(self.max_length - cur_len - 1)
                    cur_word_slices.append(cur_index)
                    cur_index = self.max_length - 1
                    break
                cur_word_lenghts.append(word_len)
                cur_word_slices.append(cur_index)
                cur_index += word_len
                cur_len += word_len
            cur_word_lenghts.append(1)
            cur_word_slices.append(cur_index)
            # while len(cur_word_lenghts) < self.max_length:
            #     cur_word_lenghts.append(0)
            #     cur_word_slices.append(-1)
            return cur_word_lenghts, cur_word_slices

        for data_ele in data_json:
            cur_word_lenghts_baidu, cur_word_slices_baidu = one_type_lac("baidu")
            cur_word_lenghts_thu, cur_word_slices_thu = one_type_lac("thu")
            cur_word_lenghts_ltp, cur_word_slices_ltp = one_type_lac("ltp")
            word_length_list_baidu.append(cur_word_lenghts_baidu)
            word_slice_list_baidu.append(cur_word_slices_baidu)
            word_length_list_thu.append(cur_word_lenghts_thu)
            word_slice_list_thu.append(cur_word_slices_thu)
            word_length_list_ltp.append(cur_word_lenghts_ltp)
            word_slice_list_ltp.append(cur_word_slices_ltp)
        return word_length_list_baidu, word_slice_list_baidu, word_length_list_thu, word_slice_list_thu, word_length_list_ltp, word_slice_list_ltp

    def load_cache_train_dev_data(self, train_flag=True, test_flag=False):
        train_data_X = np.load(self.train_X_path, allow_pickle=True)
        dev_data_X = np.load(self.valid_X_path, allow_pickle=True)
        train_data_Y = np.load(self.train_Y_path, allow_pickle=True)
        dev_data_Y = np.load(self.valid_Y_path, allow_pickle=True)
        train_word_length_list_baidu = np.load(self.train_word_seq_length_path_baidu, allow_pickle=True)
        train_word_slice_list_baidu = np.load(self.train_word_slice_path_baidu, allow_pickle=True)
        train_word_length_list_thu = np.load(self.train_word_seq_length_path_thu, allow_pickle=True)
        train_word_slice_list_thu = np.load(self.train_word_slice_path_thu, allow_pickle=True)
        train_word_length_list_ltp = np.load(self.train_word_seq_length_path_ltp, allow_pickle=True)
        train_word_slice_list_ltp = np.load(self.train_word_slice_path_ltp, allow_pickle=True)
        dev_word_length_list_baidu = np.load(self.dev_word_seq_length_path_baidu, allow_pickle=True)
        dev_word_slice_list_baidu = np.load(self.dev_word_slice_path_baidu, allow_pickle=True)
        dev_word_length_list_thu = np.load(self.dev_word_seq_length_path_thu, allow_pickle=True)
        dev_word_slice_list_thu = np.load(self.dev_word_slice_path_thu, allow_pickle=True)
        dev_word_length_list_ltp = np.load(self.dev_word_seq_length_path_ltp, allow_pickle=True)
        dev_word_slice_list_ltp = np.load(self.dev_word_slice_path_ltp, allow_pickle=True)
        train_input_lens = np.load(self.train_input_lens_path, allow_pickle=True)
        dev_input_lens = np.load(self.valid_input_lens_path, allow_pickle=True)
        train_input_masks = np.load(self.train_input_masks_path, allow_pickle=True)
        dev_input_masks = np.load(self.valid_input_masks_path, allow_pickle=True)
        train_input_segment_ids = np.load(self.train_input_segment_ids_path, allow_pickle=True)
        dev_input_segment_ids = np.load(self.valid_input_segment_ids_path, allow_pickle=True)

        test_data_X = np.load(self.test_X_path, allow_pickle=True)
        test_data_Y = np.load(self.test_Y_path, allow_pickle=True)
        test_word_length_list_baidu = np.load(self.test_word_seq_length_path_baidu, allow_pickle=True)
        test_word_slice_list_baidu = np.load(self.test_word_slice_path_baidu, allow_pickle=True)
        test_word_length_list_thu = np.load(self.test_word_seq_length_path_thu, allow_pickle=True)
        test_word_slice_list_thu = np.load(self.test_word_slice_path_thu, allow_pickle=True)
        test_word_length_list_ltp = np.load(self.test_word_seq_length_path_ltp, allow_pickle=True)
        test_word_slice_list_ltp = np.load(self.test_word_slice_path_ltp, allow_pickle=True)
        test_input_lens = np.load(self.test_input_lens_path, allow_pickle=True)
        test_input_masks = np.load(self.test_input_masks_path, allow_pickle=True)
        test_input_segment_ids = np.load(self.test_input_segment_ids_path, allow_pickle=True)

        if train_flag:
            return {"input_ids": train_data_X, "input_mask": train_input_masks, "segment_ids": train_input_segment_ids,
                    "label_ids": train_data_Y,
                    "input_lens": train_input_lens, "word_seq_len_baidu": train_word_length_list_baidu,
                    "word_seq_len_thu": train_word_length_list_thu,
                    "word_seq_len_ltp": train_word_length_list_ltp, "word_slice_baidu": train_word_slice_list_baidu,
                    "word_slice_thu": train_word_slice_list_thu,
                    "word_slice_ltp": train_word_slice_list_ltp}
        elif not test_flag:
            return {"input_ids": dev_data_X, "input_mask": dev_input_masks, "segment_ids": dev_input_segment_ids,
                    "label_ids": dev_data_Y,
                    "input_lens": dev_input_lens, "word_seq_len_baidu": dev_word_length_list_baidu,
                    "word_seq_len_thu": dev_word_length_list_thu,
                    "word_seq_len_ltp": dev_word_length_list_ltp, "word_slice_baidu": dev_word_slice_list_baidu,
                    "word_slice_thu": dev_word_slice_list_thu,
                    "word_slice_ltp": dev_word_slice_list_ltp}
        else:

            return {"input_ids": test_data_X, "input_mask": test_input_masks, "segment_ids": test_input_segment_ids,
                    "label_ids": test_data_Y,
                    "input_lens": test_input_lens, "word_seq_len_baidu": test_word_length_list_baidu,
                    "word_seq_len_thu": test_word_length_list_thu,
                    "word_seq_len_ltp": test_word_length_list_ltp, "word_slice_baidu": test_word_slice_list_baidu,
                    "word_slice_thu": test_word_slice_list_thu,
                    "word_slice_ltp": test_word_slice_list_ltp}