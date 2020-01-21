import codecs
from data_processing.basic_prepare_data import BaseDataPreparing

class bertPrepareData(BaseDataPreparing):
    def __init__(self,vocab_file,slot_file,config,bert_file,max_length,gen_new_data=False,is_inference=False,label_less=True):
        self.bert_file = bert_file
        self.max_length = max_length
        self.label_less = label_less
        super(bertPrepareData,self).__init__(vocab_file,slot_file,config,pretrained_embedding_file=None,word_embedding_file=None,load_w2v_embedding=False,load_word_embedding=False,gen_new_data=gen_new_data,is_inference=is_inference)


    def tranform_singlg_data_example(self,text):
        # print(text)
        word_list = []
        # if not self.label_less:
        #     word_list.append("[CLS]")
        word_list.append("[CLS]")

        if self.is_inference:
            word_list.extend([w for w in text if w !=" "])
        else:
            word_list.extend(self.tokenizer.tokenize(text))
        if len(word_list)>=self.max_length:
            word_list = word_list[0:self.max_length-1]
        # if not self.label_less:
        #     word_list.append("[SEP]")
        word_list.append("[SEP]")
        word_id_list = self.tokenizer.convert_tokens_to_ids(word_list)
        # print(len(word_id_list))
        return word_id_list

    def trans_orig_data_to_training_data(self,datas_file):
        data_X = []
        data_Y = []
        with codecs.open(datas_file,'r','utf-8') as fr:
            for index,line in enumerate(fr):
                line = line.strip("\n")
                if index % 2 == 0:
                    data_X.append(self.tranform_singlg_data_example(line))
                else:
                    # slot_list = ["[CLS]"]
                    slot_list = []
                    # if not self.label_less:
                    #     slot_list.append("[CLS]")

                    slot_list.append("O")

                    slot_list.extend(self.tokenizer.tokenize(line))
                    if len(slot_list) >= self.max_length:
                        slot_list = slot_list[0:self.max_length - 1]

                    # if not self.label_less:
                    #     slot_list.append("[SEP]")

                    slot_list.append("O")
                    slot_list = [slots.upper() for slots in slot_list]
                    slot_id_list = self.tokenizer.convert_slot_to_ids(slot_list)
                    data_Y.append(slot_id_list)
        return data_X,data_Y
