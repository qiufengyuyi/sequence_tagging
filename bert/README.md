[TOC]

# Use BERT as feature
1. 如何调用bert，将输入的语句输出为向量？
2. 如果在自己的代码中添加bert作为底层特征，需要官方例子run_classifier.py的那么多代码吗？
# 环境

```python
mac:
tf==1.4.0
python=2.7

windows:
tf==1.12
python=3.5
```

# 入口

调用预训练的模型，来做句子的预测。
bert_as_feature.py
配置data_root为模型的地址
调用预训练模型：chinese_L-12_H-768_A-12
调用核心代码：
```python
# graph
input_ids = tf.placeholder(tf.int32, shape=[None, None], name='input_ids')
input_mask = tf.placeholder(tf.int32, shape=[None, None], name='input_masks')
segment_ids = tf.placeholder(tf.int32, shape=[None, None], name='segment_ids')

# 初始化BERT
model = modeling.BertModel(
    config=bert_config,
    is_training=False,
    input_ids=input_ids,
    input_mask=input_mask,
    token_type_ids=segment_ids,
    use_one_hot_embeddings=False)

# 加载bert模型
tvars = tf.trainable_variables()
(assignment, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_check_point)

# 获取最后一层和倒数第二层。
encoder_last_layer = model.get_sequence_output()
encoder_last2_layer = model.all_encoder_layers[-2]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    token = tokenization.CharTokenizer(vocab_file=bert_vocab_file)
    query = u'Jack,请回答1988, UNwant\u00E9d,running'
    split_tokens = token.tokenize(query)
    word_ids = token.convert_tokens_to_ids(split_tokens)
    word_mask = [1] * len(word_ids)
    word_segment_ids = [0] * len(word_ids)
    fd = {input_ids: [word_ids], input_mask: [word_mask], segment_ids: [word_segment_ids]}
    last, last2 = sess.run([encoder_last_layer, encoder_last_layer], feed_dict=fd)
    print('last shape:{}, last2 shape: {}'.format(last.shape, last2.shape))
```

完整代码见： [bert_as_feature.py](https://github.com/InsaneLife/bert/blob/master/bert_as_feature.py) 

代码库：https://github.com/InsaneLife/bert

中文模型下载：**[`BERT-Base, Chinese`](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)**:    Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads, 110M    parameters

# 最终结果

最后一层和倒数第二层：
last shape:(1, 14, 768), last2 shape: (1, 14, 768)

```
# last value
[[ 0.8200665   1.7532703  -0.3771637  ... -0.63692784 -0.17133102
   0.01075665]
 [ 0.79148203 -0.08384223 -0.51832616 ...  0.8080162   1.9931345
   1.072408  ]
 [-0.02546642  2.2759912  -0.6004753  ... -0.88577884  3.1459959
  -0.03815675]
 ...
 [-0.15581022  1.154014   -0.96733016 ... -0.47922543  0.51068854
   0.29749477]
 [ 0.38253042  0.09779643 -0.39919692 ...  0.98277044  0.6780443
  -0.52883977]
 [ 0.20359193 -0.42314947  0.51891303 ... -0.23625426  0.666618
   0.30184716]]
```



# 预处理

`tokenization.py`是对输入的句子处理，包含两个主要类：`BasickTokenizer`, `FullTokenizer`

`BasickTokenizer`会对每个字做分割，会识别英文单词，对于数字会合并，例如：

```
query: 'Jack,请回答1988, UNwant\u00E9d,running'
token: ['jack', ',', '请', '回', '答', '1988', ',', 'unwanted', ',', 'running']
```

`FullTokenizer`会对英文字符做n-gram匹配，会将英文单词拆分，例如running会拆分为run、##ing，主要是针对英文。

```
query: 'UNwant\u00E9d,running'
token: ["un", "##want", "##ed", ",", "runn", "##ing"]
```

对于中文数据，特别是NER，如果数字和英文单词是整体的话，会出现大量UNK，所以要将其拆开，想要的结果：

```
query: 'Jack,请回答1988'
token:  ['j', 'a', 'c', 'k', ',', '请', '回', '答', '1', '9', '8', '8']
```

具体变动如下：

```python
class CharTokenizer(object):
    """Runs end-to-end tokenziation."""
    def __init__(self, vocab_file, do_lower_case=True):
        self.vocab = load_vocab(vocab_file)
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in token:
                split_tokens.append(sub_token)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_tokens_to_ids(self.vocab, tokens)
```






