import tensorflow as tf

class CNNLAYER(object):
    def __init__(self,kernel_size,kernel_nums,dropout_rate,hidden_size,is_training):
        self.kernel_size = kernel_size
        self.kernel_nums = kernel_nums
        self.dropout_rate = dropout_rate
        self.hidden_size = hidden_size
        self.is_training = is_training

    def add_conv(self,input):
        t_conv = tf.layers.conv1d(input, self.kernel_nums, self.kernel_size,strides=1, padding='same',activation=tf.nn.relu)
        t_conv = tf.layers.dropout(t_conv, rate=self.dropout_rate,training=self.is_training)
        output = tf.layers.dense(t_conv,self.hidden_size)
        return output
