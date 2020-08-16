import tensorflow as tf
import tensorflow_probability as tfp
import torch.nn as nn
nn.KLDivLoss()
# tf.enable_eager_execution()

# batch_size_tensor = tf.Variable(tf.range(0,4))
# seq_max_len_tensor = tf.range(0,3)
# batch_zeros = tf.Variable(tf.zeros((4,3,3),dtype=tf.int32))
# print(batch_zeros.get_shape().as_list())
# batch_size_tensor = tf.expand_dims(batch_size_tensor,1)
# batch_size_tensor = tf.tile(batch_size_tensor,[1,3])
# # batch_size_tensor = tf.expand_dims(batch_size_tensor,1)
# # res = tf.add(batch_size_tensor,batch_zeros)
# indices = tf.Variable([[[0]], [[0]], [[0]], [[0]]])
# # res = tf.scatter_nd_update(batch_zeros, indices, batch_size_tensor)
# seq_max_len_tensor = tf.expand_dims(seq_max_len_tensor,0)
# seq_max_len_tensor = tf.tile(seq_max_len_tensor,[4,1])
# # batch_size_tensor = tf.expand_dims(batch_size_tensor,1)
# # res = tf.add(seq_max_len_tensor,batch_size_tensor)
# const_var = tf.constant([[1,0,2],[0,0,1],[1,1,2],[2,1,0]])
# # print(batch_zeros[:,:,0])
# batch_zeros[:,:,0].assign(batch_size_tensor)
# batch_zeros[:,:,1].assign(seq_max_len_tensor)
# batch_zeros[:,:,2].assign(const_var)
# print(batch_zeros)
# seq_zeros = tf.zeros((4,3),dtype=tf.int32)
# batch_size_tensor = tf.expand_dims(batch_size_tensor,1)
# res = tf.concat([batch_size_tensor,seq_zeros],axis=1)
# print(res)
# # res = tf.tile(res,[1,3])
# # # seq_max_len_tensor = tf.expand_dims()
# # res = tf.add(res,seq_zeros)
# with tf.Session() as sess:
#     print(sess.run(res))
# # print(res)

# tf.enable_eager_execution()
#
# data = tf.Variable([[2],
#                     [3],
#                     [4],
#                     [5],
#                     [6]])
#
# cond = tf.where(tf.less(data, 5)) # update value less than 5
# match_data = tf.gather_nd(data, cond)
# square_data = tf.square(match_data) # square value less than 5
#
# data = tf.scatter_nd_update(data, cond, square_data)
#
# print(data)

def cal_binary_dsc_loss(logits,labels,seq_mask,num_labels,one_hot=True,smoothing_lambda=1.0):
    # 这里暂时不用mask，因为mask的地方，label都是0，会被忽略掉
    if one_hot:
        labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    else:
        labels = tf.expand_dims(labels,axis=-1)
    # seq_mask = tf.cast(seq_mask, tf.float32)
    predict_prob = tf.nn.softmax(logits, axis=-1, name="predict_prob")
    pos_prob = predict_prob[:, :, 1]
    neg_prob = predict_prob[:, :, 0]
    pos_label = labels[:, :, 1]
    nominator = neg_prob * pos_prob * pos_label
    denominator = neg_prob * pos_prob + pos_label
    loss = (nominator + smoothing_lambda)/(denominator + smoothing_lambda)
    loss = 1. - loss
    loss = tf.reduce_sum(loss,axis=-1)
    loss = tf.reduce_mean(loss)
    return loss

def dice_dsc_loss(logits,labels,text_length_list,seq_mask,slot_label_num,smoothing_lambda=1.0):
    """
    dice loss dsc
    :param logits: [batch_size,time_step,num_class]
    :param labels: [batch_size,time_step]
    :param seq_length:[batch_size]
    :return:
    """

    predict_prob = tf.nn.softmax(logits,axis=-1,name="predict_prob")
    label_one_hot = tf.one_hot(labels, depth=slot_label_num, axis=-1)
    # seq_mask = tf.sequence_mask(seq_mask)
    # seq_mask = tf.cast(seq_mask,dtype=tf.float32)
    # batch_size_tensor = tf.range(0,tf.shape(logits)[0])
    # seq_max_len_tensor = tf.range(0,tf.shape(logits)[1])
    # # batch_size_tensor = tf.expand_dims(batch_size_tensor,1)
    # seq_max_len_tensor = tf.expand_dims(seq_max_len_tensor,axis=0)
    # seq_max_len_tensor = tf.tile(seq_max_len_tensor,[tf.shape(logits)[0],1])
    # seq_max_len_tensor = tf.expand_dims(seq_max_len_tensor,axis=-1)
    # batch_size_tensor = tf.expand_dims(batch_size_tensor, 1)
    # batch_size_tensor = tf.tile(batch_size_tensor, [1, tf.shape(logits)[1]])
    # batch_size_tensor = tf.expand_dims(batch_size_tensor, -1)
    # # batch_zeros_result = tf.zeros((tf.shape(logits)[0],tf.shape(logits)[1],3), dtype=tf.int32)
    # labels = tf.expand_dims(labels,axis=-1)
    # gather_idx = tf.concat([batch_size_tensor,seq_max_len_tensor,labels],axis=-1)
    # gather_result = tf.gather_nd(predict_prob,gather_idx)
    # # gather_result = gather_result
    # neg_prob = 1. - gather_result
    # neg_prob = neg_prob
    # # gather_result = gather_result
    # cost = 1. - neg_prob*gather_result/(neg_prob*gather_result+1.)
    # cost = cost * seq_mask
    # cost = tf.reduce_sum(cost,axis=-1)
    # cost = tf.reduce_mean(cost)
    # return cost
    # neg_prob = 1.- predict_prob
    nominator = 2*predict_prob*label_one_hot+smoothing_lambda
    denomiator = predict_prob*predict_prob+label_one_hot*label_one_hot+smoothing_lambda
    result = nominator/denomiator
    result = 1. - result
    result = tf.reduce_sum(result,axis=-1)
    result = result * seq_mask
    result = tf.reduce_sum(result,axis=-1,keep_dims=True)
    result = result/tf.cast(text_length_list,tf.float32)
    result = tf.reduce_mean(result)
    return result
    # cost = cal_binary_dsc_loss(predict_prob[:, :, 0],label_one_hot[:, :, 0],seq_mask)
    # for i in range(1,slot_label_num):
    #     cost += cal_binary_dsc_loss(predict_prob[:, :, i],label_one_hot[:, :, i],seq_mask)
    #     # print(denominator)
    # cost = cost/float(slot_label_num)
    # cost = tf.reduce_mean(cost)
    # return cost

def vanilla_dsc_loss(logits,labels,seq_mask,num_labels,smoothing_lambda=1.0,one_hot=True):
    if one_hot:
        labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    else:
        labels = tf.expand_dims(labels,axis=-1)

    predict_prob = tf.nn.softmax(logits, axis=-1, name="predict_prob")
    pos_prob = predict_prob[:, :, 1]
    neg_prob = predict_prob[:,:,0]
    pos_label = labels[:, :, 1]
    neg_label = labels[:,:,0]
    denominator = 2 * pos_prob * pos_label + neg_prob * pos_label + pos_prob * neg_label + smoothing_lambda
    nominator = 2 * pos_prob * pos_label + smoothing_lambda
    loss = 1. - nominator / denominator
    loss = loss * tf.cast(seq_mask,tf.float32)
    loss = tf.reduce_sum(loss,axis=-1)
    loss = tf.reduce_mean(loss,axis=0)
    return loss

def dl_dsc_loss(logits,labels,text_length_list,seq_mask,slot_label_num,smoothing_lambda=1.0,gamma=2.0):
    predict_prob = tf.nn.softmax(logits, axis=-1, name="predict_prob")
    label_one_hot = tf.one_hot(labels, depth=slot_label_num, axis=-1)
    # neg_prob = 1.- predict_prob
    # neg_prob = tf.pow(neg_prob,gamma)
    pos_prob = predict_prob[:,:,1]
    pos_prob_squre = tf.pow(pos_prob,2)
    pos_label = label_one_hot[:,:,1]
    pos_label_squre = tf.pow(pos_label,2)
    nominator = 2*pos_prob_squre*pos_label_squre+smoothing_lambda
    denominator = pos_label_squre+pos_label_squre+smoothing_lambda
    result = nominator/denominator
    result = 1.-result
    result = result * tf.cast(seq_mask,tf.float32)
    result = tf.reduce_sum(result, axis=-1, keep_dims=True)
    result = result / tf.cast(text_length_list, tf.float32)
    result = tf.reduce_mean(result)
    return result

def ce_loss(logits,labels,mask,num_labels,one_hot=True,imbalanced_ratio=2):
    if one_hot:
        labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    else:
        labels = tf.expand_dims(labels,axis=-1)
    probs = tf.nn.softmax(logits,axis=-1)
    pos_probs = probs[:,:,1]
    pos_probs = tf.pow(pos_probs,imbalanced_ratio)
    pos_probs = tf.expand_dims(pos_probs, axis=-1)
    neg_probs = 1. - pos_probs
    probs = tf.concat([neg_probs,pos_probs],axis=-1)
    print(probs)
    log_probs = tf.log(probs+1e-7)
    per_example_loss = -tf.reduce_sum(tf.cast(labels,tf.float32) * log_probs, axis=-1)
    per_example_loss = per_example_loss * tf.cast(mask, tf.float32)
    loss = tf.reduce_sum(per_example_loss, axis=-1)
    loss = tf.reduce_mean(loss)
    return loss

def focal_loss(logits,labels,mask,num_labels,one_hot=True,lambda_param=1.5):
    probs = tf.nn.softmax(logits,axis=-1)
    pos_probs = probs[:,:,1]
    prob_label_pos = tf.where(tf.equal(labels,1),pos_probs,tf.ones_like(pos_probs))
    prob_label_neg = tf.where(tf.equal(labels,0),pos_probs,tf.zeros_like(pos_probs))
    loss = tf.pow(1. - prob_label_pos,lambda_param)*tf.log(prob_label_pos + 1e-7) + \
           tf.pow(prob_label_neg,lambda_param)*tf.log(1. - prob_label_neg + 1e-7)
    loss = -loss * tf.cast(mask,tf.float32)
    loss = tf.reduce_sum(loss,axis=-1,keepdims=True)
    # loss = loss/tf.cast(tf.reduce_sum(mask,axis=-1),tf.float32)
    loss = tf.reduce_mean(loss)
    return loss

def span_loss(logits,labels,mask):
    probs = tf.nn.softmax(logits,axis=1)
    arg_max_label = tf.cast(tf.where(probs > 0.5,tf.ones_like(labels),tf.zeros_like(labels)),tf.int32)
    arg_max_label *= mask


def test(seq_length,all_length):
    mask = tf.cast(tf.sequence_mask(seq_length),tf.int32)
    mask = mask * -1
    left = tf.zeros((4,3),dtype=tf.int32)
    all = tf.concat((mask,left),axis=-1)
    all_mask = tf.cast(tf.sequence_mask(all_length),tf.int32)
    all = all + all_mask
    return all
def test2():
    const_var = tf.constant([[0,0,0,1,0,1], [0,0, 0, 1,0,0], [0,0,1,0,0,0], [0, 1, 0,1,1,0]])
    a = tf.expand_dims(const_var,axis=-1)
    b = tf.where(a)
    return b

if __name__ == "__main__":
    # const_var = tf.constant([[0,0,0,1,0,1], [0,0, 0, 1,0,0], [0,0,1,0,0,0], [0, 1, 0,1,1,0]])
    # logits_tensor = tf.Variable([[[0.2,0.4],[0.1,0.6],[0.88,0.4],[0.2,0.4],[0.1,0.6],[0.88,0.4]],[[0.2,0.4],[0.4,0.6],[0.4,23],[0.2,0.4],[0.4,0.6],[0.4,23]],[[0.1,0.2],[0.1,0.4],[0.88,0.4],[0.1,0.2],[0.1,0.4],[0.88,0.4]],[[0.1,0.4],[0.4,0.6],[0.88,23],[0.1,0.4],[0.4,0.6],[0.88,23]]])
    # seq_length_s = tf.Variable([3,1,2,1])
    # all_length_s = tf.Variable([6,4,3,5])
    # mask_all = test(seq_length_s,all_length_s)
    # loss,labels = ce_loss(logits_tensor,const_var,mask_all,2)

    result = test2()


    # cost,b = focal_dsc_loss(logits_tensor,const_var,seq_length_s,"",3)
    # print(selected_probs)
    # print(idx)
        # gathered_prob = tf.gather_nd(predict_prob)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # print(sess.run(orig_prob))
        # print(sess.run(idx))
        # print(sess.run(selected_probs))
        print(sess.run(result))
