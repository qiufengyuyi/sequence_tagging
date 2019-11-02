# sequence tagging project

依赖包：主要是tensorflow 1.12.0，其余见requirements.txt

目前项目包含了传统的Bilstm-crf模型和使用了bert的模型。

针对的数据：目前是基于字符级别标注的实体识别数据。使用网上公开的字符级的中文词向量。

TODO：

1、结合词级别的词向量与字符向量结合，做字符级别的tagging

2、joint learning with intent classification

3、不用bert，在lstm-crf基础上进行优化，增加cnn的架构，或者attention机制。

4、seq label 转化为阅读理解问题。参考最新的论文 A Unified MRC Framework for Named Entity Recognition

