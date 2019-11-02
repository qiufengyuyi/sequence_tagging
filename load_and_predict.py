import tensorflow as tf
import re
import os
import numpy as np
import codecs
from configs.base_config import config
from tensorflow.contrib import predictor
from data_processing.basic_prepare_data import BaseDataPreparing
from data_processing.bert_prepare_data import bertPrepareData
# from standard_slot_descp import *
from pathlib import Path
from sklearn.metrics import f1_score,accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
class fastPredict(object):
    def __init__(self,model_path):
        self.model_path = model_path
        self.data_loader = self.init_data_loader()
        self.predict_fn = self.load_models()

    def init_data_loader(self):
        vocab_file_path = os.path.join(config.get("embedding_dir"), config.get("vocab_file"))
        pretrained_embedding_path = os.path.join(config.get("embedding_dir"), config.get("embedding_file_name"))
        input_embedding_path = os.path.join(config.get("embedding_dir"), config.get("input_embedding_file"))
        slot_file = os.path.join(config.get("slot_list_root_path"), config.get("slot_file_name"))

        data_loader = BaseDataPreparing(vocab_file_path, slot_file, pretrained_embedding_path, input_embedding_path,
                                        False, is_inference=True)
        return data_loader

    def load_models(self):
        subdirs = [x for x in Path(self.model_path).iterdir()
                   if x.is_dir() and 'temp' not in str(x)]
        latest = str(sorted(subdirs)[-1])
        predict_fn = predictor.from_saved_model(latest)
        return predict_fn

    def predict_text(self,text,orig_labels,raw_test_data=True,using_bert=False,return_raw_result=False):
        if raw_test_data:
            text = re.sub(" ", "", text)
            words = self.data_loader.tranform_singlg_data_example(text)
        else:
            words = text
        nwords = len(words)
        predictions = self.predict_fn({'words': [words], 'text_length': [nwords]})
        label_list = predictions["output"][0]
        if using_bert:
            label_list = [label for label in label_list if label not in
                          [8,9]]

            if len(label_list) < nwords:
                # bert 截断了

                pad_list = [self.data_loader.tokenizer.slot2id["O"]]*(len(orig_labels)-len(label_list))
                label_list.extend(pad_list)
                print(label_list)
        if return_raw_result:
            return label_list
        else:
            entity_list = self.data_loader.translate_id_2_slot(text,label_list)
            return entity_list

    # def output_standard_des(self,entity_list):
    #     output_des = entrance(entity_list)
    #     return output_des

class fastPredictionBert(fastPredict):
    def __init__(self,model_path):
        self.orig_test_file = os.path.join(config.get("data_dir"), config.get("orig_test"))
        super(fastPredictionBert,self).__init__(model_path)

    def get_orig_test_label(self):
        label_lists = []
        with codecs.open(self.orig_test_file,'r','utf-8') as fr:
            for i,line in enumerate(fr):
                if i % 2 == 1:
                    line = line.strip("\n")
                    # line = re.sub(" ", "", line)
                    labels = self.data_loader.tokenizer.tokenize(line)
                    slot_list = [slots.upper() for slots in labels]
                    slot_id_list = self.data_loader.tokenizer.convert_slot_to_ids(slot_list)
                    # labels = [label for label in line]
                    label_lists.append(slot_id_list)
        return label_lists

    def init_data_loader(self):
        vocab_file_path = os.path.join(config.get("bert_pretrained_model_path"), config.get("vocab_file"))
        slot_file = os.path.join(config.get("slot_list_root_path"), config.get("slot_file_name"))
        bert_config_file = os.path.join(config.get("bert_pretrained_model_path"), config.get("bert_config_path"))
        data_loader = bertPrepareData(vocab_file_path,slot_file,bert_config_file,384,gen_new_data=False,is_inference=True)
        return data_loader

    def load_models(self):
        subdirs = [x for x in Path(self.model_path).iterdir()
                   if x.is_dir() and 'temp' not in str(x)]
        latest = str(sorted(subdirs)[-1])
        predict_fn = predictor.from_saved_model(latest)
        return predict_fn
# def load_and_predict(text):
#     text = re.sub(" ","",text)
#     vocab_file_path = os.path.join(config.get("embedding_dir"), config.get("vocab_file"))
#     pretrained_embedding_path = os.path.join(config.get("embedding_dir"), config.get("embedding_file_name"))
#     input_embedding_path = os.path.join(config.get("embedding_dir"), config.get("input_embedding_file"))
#     slot_file = os.path.join(config.get("slot_list_root_path"), config.get("slot_file_name"))
#
#     data_loader = BaseDataPreparing(vocab_file_path, slot_file, pretrained_embedding_path, input_embedding_path,
#                                     False,is_inference=True)
#     words = data_loader.tranform_singlg_data_example(text)
#     model_path = config.get("pb_model_dir")
#     subdirs = [x for x in Path(model_path).iterdir()
#                if x.is_dir() and 'temp' not in str(x)]
#     latest = str(sorted(subdirs)[-1])
#     predict_fn = predictor.from_saved_model(latest)
#     nwords = len(words)
#     predictions = predict_fn({'words': [words], 'text_length': [nwords]})
#     print(predictions["output"])

if __name__ == '__main__':
#     text_list = ["我 们 变 而 以 书 会 友 ， 以 书 结 缘 ， 把 欧 美 、 港 台 流 行 的 食 品 类 图 谱 、 画 册 、 工 具 书 汇 集 一 堂 ",
#                  "为 了 跟 踪 国 际 最 新 食 品 工 艺 、 流 行 趋 势 ， 大 量 搜 集 海 外 专 业 书 刊 资 料 是 提 高 技 艺 的 捷 径 。 其 中 线 装 古 籍 逾 千 册 ； 民 国 出 版 物 几 百 种 ； 珍 本 四 册 、 稀 见 本 四 百 余 册 ， 出 版 时 间 跨 越 三 百 余 年 。 有 的 古 木 交 柯 ， 春 机 荣 欣 ， 从 诗 人 句 中 得 之 ， 而 入 画 中 ， 观 之 令 人 心 驰 。 不 过 重 在 晋 趣 ， 略 增 明 人 气 息 ， 妙 在 集 古 有 道 、 不 露 痕 迹 罢 了 。 其 实 非 汉 非 唐 ， 又 是 什 么 与 什 么 呢 ？ 国 正 学 长 的 文 章 与 诗 词 ， 早 就 读 过 一 些 ， 很 是 喜 欢 。",
#                  "刚 建 成 的 山 西 省 最 大 的 古 玩 交 易 市 场 将 举 行 开 业 剪 彩 仪 式 ， 同 时 还 展 示 长 度 为 六 百 六 十 六 点 六 米 、 创 造 吉 尼 斯 世 界 纪 录 的 中 华 巨 龙 ， 并 推 出 一 百 零 八 种 地 方 风 味 小 吃 ， 令 嘉 宾 旅 客 切 身 感 受 平 遥 古 城 文 化 的 独 特 魅 力 。"]
#

    # fp = fastPredictionBert(config.get("bert_model_pb"))
    # orig_label_list = fp.get_orig_test_label()
    # test_data_X = np.load(fp.data_loader.test_X_path)
    # len_true = len(orig_label_list)
    # orig_label_array = np.array(orig_label_list)
    # np.save("true_label.npy",orig_label_array)
    # prediction_list = []
    # for i,text in enumerate(test_data_X):
    #     prediction_list.append(fp.predict_text(text,orig_label_list[i],raw_test_data=False,using_bert=True,return_raw_result=True))
    # # prediction_array = np.load("prediction_result.npy")
    # prediction_array = np.array(prediction_list)
    # np.save("prediction_result.npy",np.array(prediction_list))
    # # print(accuracy_score(orig_label_list,prediction_list))
    # all = np.concatenate((np.array(orig_label_list),prediction_array),axis=0)
    # mlb = MultiLabelBinarizer()
    # all_trans = mlb.fit_transform(all)
    # true_labels = all_trans[0:len_true]
    # preds = all_trans[len_true:]
    # print(f1_score(true_labels,preds,average="micro"))
    # print(f1_score())
    fp = fastPredict(config.get("pb_model_dir"))
    test_data_X = np.load(fp.data_loader.test_X_path)
    prediction_list = []
    for i,text in enumerate(test_data_X):
        prediction_list.append(fp.predict_text(text,None,raw_test_data=False,using_bert=False,return_raw_result=True))
    prediction_array = np.array(prediction_list)
    np.save("prediction_result_no_bert.npy",np.array(prediction_list))
    # print(accuracy_score(orig_label_list,prediction_list))
    true_labels = np.load("true_label.npy")
    len_true = len(true_labels)
    all = np.concatenate((true_labels,prediction_array),axis=0)
    mlb = MultiLabelBinarizer()
    all_trans = mlb.fit_transform(all)
    true_labels = all_trans[0:len_true]
    preds = all_trans[len_true:]
    print(f1_score(true_labels,preds,average="micro"))
    # print(f1_score())
    print(accuracy_score(true_labels,preds))