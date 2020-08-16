import os
import codecs
import re
from pyltp import Segmentor
import thulac
import json
from LAC import LAC
from data_processing.mrc_query_map import en_conll03_ner

# 装载分词模型
lac = LAC(mode='seg')
segmentor = Segmentor()
segmentor.load(os.path.join("ltp_data_v3.4.0", "cws.model"))
thu1 = thulac.thulac()  # 默认模式


def ltp_seg(text):
    words = segmentor.segment(text)
    return [word for word in words]


def thulac_seg(text):
    # thu1 = thulac.thulac()  #只进行分词，不进行词性标注
    word_seg = thu1.cut(text, text=False)
    word_seg = [ele[0] for ele in word_seg]
    return word_seg


def baidulac_seg(text):
    seg_result = lac.run(text)
    return seg_result


def seg_word_for_data(file_path, dest_file_path):
    seg_result_list = []
    with codecs.open(file_path, 'r', 'utf-8') as fr:
        for index, line in enumerate(fr):
            if index % 2 == 0:
                line = re.sub(r" ", "", line)
                line = line.strip("\n")
                baidu_seg = baidulac_seg(line)
                thu_seg = thulac_seg(line)
                ltp_seg_res = ltp_seg(line)
                seg_result_list.append({"baidu": baidu_seg, "thu": thu_seg, "ltp": ltp_seg_res})
    with codecs.open(dest_file_path, 'w', 'utf-8') as fw:
        json.dump(seg_result_list, fw, ensure_ascii=False)


def seg_word_for_query(dest_file_path):
    tags_list = en_conll03_ner.get("tags")
    natural_query_list = en_conll03_ner.get("natural_query")
    seg_result_dict = {}
    for tag in tags_list:
        natural_query = natural_query_list.get(tag)
        baidu_seg = baidulac_seg(natural_query)
        thu_seg = thulac_seg(natural_query)
        ltp_seg_res = ltp_seg(natural_query)
        seg_result_dict.update({tag: {"baidu": baidu_seg, "thu": thu_seg, "ltp": ltp_seg_res}})
    # for data in en_conll03_ner:
    with codecs.open(dest_file_path, 'w', 'utf-8') as fw:
        json.dump(seg_result_dict, fw, ensure_ascii=False)


if __name__ == "__main__":
    # seg_word_for_data("data/orig_data_train.txt","data/seg_orig_data_train.json")
    # seg_word_for_data("data/orig_data_dev.txt","data/seg_orig_data_dev.json")
    # seg_word_for_data("data/orig_data_test.txt","data/seg_orig_data_test.json")
    seg_word_for_query("data/seg_query_data.json")