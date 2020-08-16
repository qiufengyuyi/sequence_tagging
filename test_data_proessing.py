import os
import numpy as np
from configs.bert_word_config import bert_config
from data_processing.bert_word_prepare_data import bertWordPrepareData

def gen_mwa_bert_data():
    vocab_file_path = os.path.join(bert_config.get("bert_pretrained_model_path"), bert_config.get("vocab_file"))
    bert_config_file = os.path.join(bert_config.get("bert_pretrained_model_path"), bert_config.get("bert_config_path"))
    slot_file = os.path.join(bert_config.get("slot_list_root_path"), bert_config.get("bert_slot_complete_file_name"))
    data_loader = bertWordPrepareData(vocab_file_path,slot_file,bert_config,None,384,None,None,True,False,False)


if __name__ == "__main__":
    gen_mwa_bert_data()