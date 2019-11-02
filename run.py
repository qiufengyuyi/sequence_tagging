import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ERROR
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from argparse import ArgumentParser
from train_helper import run_train,run_bert
import numpy as np
np.set_printoptions(threshold=np.inf)
tf.logging.set_verbosity(tf.logging.INFO)

def main():
    parser = ArgumentParser()
    parser.add_argument("--model_type",default='lstm_crf',type=str)
    parser.add_argument("--dropout_prob",default=0.4,type=float)
    parser.add_argument("--rnn_units",default=128,type=int)
    parser.add_argument("--epochs",default=10,type=int)
    parser.add_argument("--lr",default=1e-5,type=float)
    parser.add_argument("--clip_norm",default=5.0,type=float)
    parser.add_argument("--train_batch_size",default=8,type=int)
    parser.add_argument("--valid_batch_size",default=8,type=int)
    parser.add_argument("--shuffle_buffer",default=128,type=int)
    parser.add_argument("--do_train",action='store_true',default=True)
    parser.add_argument("--do_test",action='store_true',default=True)
    parser.add_argument("--gen_new_data",default=False)
    parser.add_argument("--tolerant_steps",default=100,type=int)
    parser.add_argument("--run_hook_steps",default=100,type=int)
    parser.add_argument("--num_layers",default=1,type=int)
    parser.add_argument("--hidden_units",default=128,type=int)
    parser.add_argument("--print_log_steps",default=20,type=int)
    parser.add_argument("--decay_epoch",default=8,type=int)
    parser.add_argument("--pre_buffer_size",default=1,type=int)
    parser.add_argument("--bert_used",default=True,action='store_true')
    parser.add_argument("--gpu_nums",default=2,type=int)
    args = parser.parse_args()
    if args.bert_used:
        run_bert(args)
    else:
        run_train(args)



if __name__ == '__main__':
    main()