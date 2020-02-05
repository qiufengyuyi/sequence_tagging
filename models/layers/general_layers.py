import tensorflow as tf
import math
import six

seed = 20


class LayerNorm(tf.keras.layers.Layer):
    def __init__(self, params):
        super().__init__()
        self._epsilon = 1e-6
        self._units = 2 * params['rnn_units']

    def build(self, input_shape):
        self.scale = self.add_weight(name='scale',
                                     shape=[self._units],
                                     initializer=tf.ones_initializer(),
                                     trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=[self._units],
                                    initializer=tf.zeros_initializer(),
                                    trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
        norm_x = (inputs - mean) * tf.math.rsqrt(variance + self._epsilon)
        return norm_x * self.scale + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape


def get_rnn_cell(rnn_size, dropout_rate):
    decoder_cell = tf.contrib.rnn.LSTMCell(rnn_size,
                                           initializer=create_initializer())
    decoder_cell = tf.contrib.rnn.DropoutWrapper(
        cell=decoder_cell, input_keep_prob=dropout_rate)
    # decoder_cell = tf.contrib.rnn.ResidualWrapper(
    # decoder_cell, residual_fn=gnmt_residual_fn)
    return decoder_cell


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

    Args:
      tensor: A tf.Tensor to check the rank of.
      expected_rank: Python integer or list of integers, expected rank.
      name: Optional name of the tensor for the error message.

    Raises:
      ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
      tensor: A tf.Tensor object to find the shape of.
      expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
      name: Optional name of the tensor for the error message.

    Returns:
      A list of dimensions of the shape of tensor. All static dimensions will
      be returned as python integers, and dynamic dimensions will be returned
      as tf.Tensor scalars.
    """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def reshape_to_matrix(input_tensor):
    """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
    ndims = input_tensor.shape.ndims
    if ndims < 2:
        raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                         (input_tensor.shape))
    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)


def dropout(input_tensor, dropout_prob):
    """Perform dropout.

    Args:
      input_tensor: float Tensor.
      dropout_prob: Python float. The probability of dropping out a value (NOT of
        *keeping* a dimension as in `tf.nn.dropout`).

    Returns:
      A version of `input_tensor` with dropout applied.
    """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
    return output


def create_attention_mask_from_input_mask(from_tensor, to_mask):
    """Create 3D attention mask from a 2D tensor mask.

    Args:
      from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
      to_mask: int32 Tensor of shape [batch_size, to_seq_length].

    Returns:
      float Tensor of shape [batch_size, from_seq_length, to_seq_length].
    """
    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]

    to_shape = get_shape_list(to_mask, expected_rank=2)
    to_seq_length = to_shape[1]

    to_mask = tf.cast(
        tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

    # We don't assume that `from_tensor` is a mask (although it could be). We
    # don't actually care if we attend *from* padding tokens (only *to* padding)
    # tokens so we create a tensor of all ones.
    #
    # `broadcast_ones` = [batch_size, from_seq_length, 1]
    broadcast_ones = tf.ones(
        shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

    # Here we broadcast along two dimensions to create the mask.
    mask = broadcast_ones * to_mask

    return mask


# def attention_layer(from_tensor,
#                     to_tensor,
#                     attention_mask=None,
#                     num_attention_heads=1,
#                     size_per_head=512,
#                     query_act=tf.nn.relu,
#                     key_act=tf.nn.relu,
#                     value_act=tf.nn.relu,
#                     attention_probs_dropout_prob=0.0,
#                     initializer_range=0.02,
#                     do_return_2d_tensor=False,
#                     batch_size=None,
#                     from_seq_length=None,
#                     to_seq_length=None,
#                     return_alignment=False,
#                     label_nums=None):
#     """Performs multi-headed attention from `from_tensor` to `to_tensor`.
#
#     This is an implementation of multi-headed attention based on "Attention
#     is all you Need". If `from_tensor` and `to_tensor` are the same, then
#     this is self-attention. Each timestep in `from_tensor` attends to the
#     corresponding sequence in `to_tensor`, and returns a fixed-with vector.
#
#     This function first projects `from_tensor` into a "query" tensor and
#     `to_tensor` into "key" and "value" tensors. These are (effectively) a list
#     of tensors of length `num_attention_heads`, where each tensor is of shape
#     [batch_size, seq_length, size_per_head].
#
#     Then, the query and key tensors are dot-producted and scaled. These are
#     softmaxed to obtain attention probabilities. The value tensors are then
#     interpolated by these probabilities, then concatenated back to a single
#     tensor and returned.
#
#     In practice, the multi-headed attention are done with transposes and
#     reshapes rather than actual separate tensors.
#
#     Args:
#       from_tensor: float Tensor of shape [batch_size, from_seq_length,
#         from_width].
#       to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
#       attention_mask: (optional) int32 Tensor of shape [batch_size,
#         from_seq_length, to_seq_length]. The values should be 1 or 0. The
#         attention scores will effectively be set to -infinity for any positions in
#         the mask that are 0, and will be unchanged for positions that are 1.
#       num_attention_heads: int. Number of attention heads.
#       size_per_head: int. Size of each attention head.
#       query_act: (optional) Activation function for the query transform.
#       key_act: (optional) Activation function for the key transform.
#       value_act: (optional) Activation function for the value transform.
#       attention_probs_dropout_prob: (optional) float. Dropout probability of the
#         attention probabilities.
#       initializer_range: float. Range of the weight initializer.
#       do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
#         * from_seq_length, num_attention_heads * size_per_head]. If False, the
#         output will be of shape [batch_size, from_seq_length, num_attention_heads
#         * size_per_head].
#       batch_size: (Optional) int. If the input is 2D, this might be the batch size
#         of the 3D version of the `from_tensor` and `to_tensor`.
#       from_seq_length: (Optional) If the input is 2D, this might be the seq length
#         of the 3D version of the `from_tensor`.
#       to_seq_length: (Optional) If the input is 2D, this might be the seq length
#         of the 3D version of the `to_tensor`.
#
#     Returns:
#       float Tensor of shape [batch_size, from_seq_length,
#         num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
#         true, this will be of shape [batch_size * from_seq_length,
#         num_attention_heads * size_per_head]).
#
#     Raises:
#       ValueError: Any of the arguments or tensor shapes are invalid.
#     """
#
#     def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
#                              seq_length, width):
#         output_tensor = tf.reshape(
#             input_tensor, [batch_size, seq_length, num_attention_heads, width])
#
#         output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
#         return output_tensor
#
#     from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
#     to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])
#
#     if len(from_shape) != len(to_shape):
#         raise ValueError(
#             "The rank of `from_tensor` must match the rank of `to_tensor`.")
#
#     if len(from_shape) == 3:
#         batch_size = from_shape[0]
#         from_seq_length = from_shape[1]
#         to_seq_length = to_shape[1]
#     elif len(from_shape) == 2:
#         if (batch_size is None or from_seq_length is None or to_seq_length is None):
#             raise ValueError(
#                 "When passing in rank 2 tensors to attention_layer, the values "
#                 "for `batch_size`, `from_seq_length`, and `to_seq_length` "
#                 "must all be specified.")
#
#     # Scalar dimensions referenced here:
#     #   B = batch size (number of sequences)
#     #   F = `from_tensor` sequence length
#     #   T = `to_tensor` sequence length
#     #   N = `num_attention_heads`
#     #   H = `size_per_head`
#
#     from_tensor_2d = reshape_to_matrix(from_tensor)
#     to_tensor_2d = reshape_to_matrix(to_tensor)
#
#     # `query_layer` = [B*F, N*H]
#     query_layer = tf.layers.dense(
#         from_tensor_2d,
#         num_attention_heads * size_per_head,
#         activation=query_act,
#         name="query",
#         kernel_initializer=create_initializer(initializer_range))
#
#     # `key_layer` = [B*T, N*H]
#     key_layer = tf.layers.dense(
#         to_tensor_2d,
#         num_attention_heads * size_per_head,
#         activation=key_act,
#         name="key",
#         kernel_initializer=create_initializer(initializer_range))
#
#     # `value_layer` = [B*T, N*H]
#     value_layer = tf.layers.dense(
#         to_tensor_2d,
#         num_attention_heads * size_per_head,
#         activation=value_act,
#         name="value",
#         kernel_initializer=create_initializer(initializer_range))
#
#     # `query_layer` = [B, N, F, H]
#     query_layer = transpose_for_scores(query_layer, batch_size,
#                                        num_attention_heads, from_seq_length,
#                                        size_per_head)
#
#     # `key_layer` = [B, N, T, H]
#     key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
#                                      to_seq_length, size_per_head)
#
#     # Take the dot product between "query" and "key" to get the raw
#     # attention scores.
#     # `attention_scores` = [B, N, F, T]
#     attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
#     attention_scores = tf.multiply(attention_scores,
#                                    1.0 / math.sqrt(float(size_per_head)))
#
#     # if attention_mask is None:
#     #     # `attention_mask` = [B, 1, F, T]
#     #     # reference original code
#     #     attention_mask = tf.sign(tf.abs(tf.reduce_sum(to_tensor,axis=-1)))
#     #     attention_mask = tf.expand_dims(attention_mask, axis=1)
#     #     attention_mask = tf.expand_dims(attention_mask, axis=1)
#     #     attention_mask = tf.tile(attention_mask,[1,num_attention_heads,from_seq_length,1])
#     #     # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
#     #     # masked positions, this operation will create a tensor which is 0.0 for
#     #     # positions we want to attend and -10000.0 for masked positions.
#     #     adder = (1.0 - tf.cast(attention_mask, tf.float32)) * tf.float32.min
#     #     # adder = tf.tile(adder,[])
#     #     # Since we are adding it to the raw scores before the softmax, this is
#     #     # effectively the same as removing these entirely.
#     #     attention_scores = tf.add(attention_scores,adder)
#     #
#     # # Normalize the attention scores to probabilities.
#     # # `attention_probs` = [B, N, F, T]
#     #
#     # query_masks = tf.sign(tf.abs(tf.reduce_sum(from_tensor,axis=-1)))
#     # query_masks = tf.expand_dims(query_masks, 1)
#     # query_masks = tf.expand_dims(query_masks, -1)
#     # query_masks = tf.tile(query_masks,[1,num_attention_heads,1,to_seq_length])
#     if return_alignment:
#         # attention_scores = attention_scores * query_masks
#         # attention_scores = dropout(attention_scores, attention_probs_dropout_prob)
#         attention_scores = tf.reshape(attention_scores,[batch_size,from_seq_length,to_seq_length*num_attention_heads])
#         # attention_scores = tf.layers.dense(
#         #     attention_scores,label_nums,
#         #     activation=key_act,name="alignment_dense",
#         #     kernel_initializer=create_initializer(initializer_range))
#         return attention_scores
#     else:
#         attention_probs = tf.nn.softmax(attention_scores)
#         attention_probs = attention_probs * query_masks
#         attention_probs = dropout(attention_probs, attention_probs_dropout_prob)
#         # `value_layer` = [B, T, N, H]
#         value_layer = tf.reshape(
#             value_layer,
#             [batch_size, to_seq_length, num_attention_heads, size_per_head])
#
#         # `value_layer` = [B, N, T, H]
#         value_layer = tf.transpose(value_layer, [0, 2, 1, 3])
#
#         # `context_layer` = [B, N, F, H]
#         context_layer = tf.matmul(attention_probs, value_layer)
#
#         # `context_layer` = [B, F, N, H]
#         context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
#
#         if do_return_2d_tensor:
#             # `context_layer` = [B*F, N*V]
#             context_layer = tf.reshape(
#                 context_layer,
#                 [batch_size * from_seq_length, num_attention_heads * size_per_head])
#         else:
#             # `context_layer` = [B, F, N*V]
#             context_layer = tf.reshape(
#                 context_layer,
#                 [batch_size, from_seq_length, num_attention_heads * size_per_head])
#         context_layer += from_tensor
#         return context_layer

def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None):
  """Performs multi-headed attention from `from_tensor` to `to_tensor`.

  This is an implementation of multi-headed attention based on "Attention
  is all you Need". If `from_tensor` and `to_tensor` are the same, then
  this is self-attention. Each timestep in `from_tensor` attends to the
  corresponding sequence in `to_tensor`, and returns a fixed-with vector.

  This function first projects `from_tensor` into a "query" tensor and
  `to_tensor` into "key" and "value" tensors. These are (effectively) a list
  of tensors of length `num_attention_heads`, where each tensor is of shape
  [batch_size, seq_length, size_per_head].

  Then, the query and key tensors are dot-producted and scaled. These are
  softmaxed to obtain attention probabilities. The value tensors are then
  interpolated by these probabilities, then concatenated back to a single
  tensor and returned.

  In practice, the multi-headed attention are done with transposes and
  reshapes rather than actual separate tensors.

  Args:
    from_tensor: float Tensor of shape [batch_size, from_seq_length,
      from_width].
    to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
    attention_mask: (optional) int32 Tensor of shape [batch_size,
      from_seq_length, to_seq_length]. The values should be 1 or 0. The
      attention scores will effectively be set to -infinity for any positions in
      the mask that are 0, and will be unchanged for positions that are 1.
    num_attention_heads: int. Number of attention heads.
    size_per_head: int. Size of each attention head.
    query_act: (optional) Activation function for the query transform.
    key_act: (optional) Activation function for the key transform.
    value_act: (optional) Activation function for the value transform.
    attention_probs_dropout_prob: (optional) float. Dropout probability of the
      attention probabilities.
    initializer_range: float. Range of the weight initializer.
    do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
      * from_seq_length, num_attention_heads * size_per_head]. If False, the
      output will be of shape [batch_size, from_seq_length, num_attention_heads
      * size_per_head].
    batch_size: (Optional) int. If the input is 2D, this might be the batch size
      of the 3D version of the `from_tensor` and `to_tensor`.
    from_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `from_tensor`.
    to_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `to_tensor`.

  Returns:
    float Tensor of shape [batch_size, from_seq_length,
      num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
      true, this will be of shape [batch_size * from_seq_length,
      num_attention_heads * size_per_head]).

  Raises:
    ValueError: Any of the arguments or tensor shapes are invalid.
  """

  def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                           seq_length, width):
    output_tensor = tf.reshape(
        input_tensor, [batch_size, seq_length, num_attention_heads, width])

    output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
    return output_tensor

  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
  to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

  if len(from_shape) != len(to_shape):
    raise ValueError(
        "The rank of `from_tensor` must match the rank of `to_tensor`.")

  if len(from_shape) == 3:
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]
    to_seq_length = to_shape[1]
  elif len(from_shape) == 2:
    if (batch_size is None or from_seq_length is None or to_seq_length is None):
      raise ValueError(
          "When passing in rank 2 tensors to attention_layer, the values "
          "for `batch_size`, `from_seq_length`, and `to_seq_length` "
          "must all be specified.")

  # Scalar dimensions referenced here:
  #   B = batch size (number of sequences)
  #   F = `from_tensor` sequence length
  #   T = `to_tensor` sequence length
  #   N = `num_attention_heads`
  #   H = `size_per_head`

  from_tensor_2d = reshape_to_matrix(from_tensor)
  to_tensor_2d = reshape_to_matrix(to_tensor)

  # `query_layer` = [B*F, N*H]
  query_layer = tf.layers.dense(
      from_tensor_2d,
      num_attention_heads * size_per_head,
      activation=query_act,
      name="query",
      kernel_initializer=create_initializer(initializer_range))

  # `key_layer` = [B*T, N*H]
  key_layer = tf.layers.dense(
      to_tensor_2d,
      num_attention_heads * size_per_head,
      activation=key_act,
      name="key",
      kernel_initializer=create_initializer(initializer_range))

  # `value_layer` = [B*T, N*H]
  value_layer = tf.layers.dense(
      to_tensor_2d,
      num_attention_heads * size_per_head,
      activation=value_act,
      name="value",
      kernel_initializer=create_initializer(initializer_range))

  # `query_layer` = [B, N, F, H]
  query_layer = transpose_for_scores(query_layer, batch_size,
                                     num_attention_heads, from_seq_length,
                                     size_per_head)

  # `key_layer` = [B, N, T, H]
  key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                   to_seq_length, size_per_head)

  # Take the dot product between "query" and "key" to get the raw
  # attention scores.
  # `attention_scores` = [B, N, F, T]
  attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
  # no scaling
  # attention_scores = tf.multiply(attention_scores,
  #                                1.0 / math.sqrt(float(size_per_head)))

  if attention_mask is not None:
    # `attention_mask` = [B, 1, F, T]
    attention_mask = tf.expand_dims(attention_mask, axis=[1])

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    attention_scores += adder

  # Normalize the attention scores to probabilities.
  # `attention_probs` = [B, N, F, T]
  attention_probs = tf.nn.softmax(attention_scores)

  # This is actually dropping out entire tokens to attend to, which might
  # seem a bit unusual, but is taken from the original Transformer paper.
  attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

  # `value_layer` = [B, T, N, H]
  value_layer = tf.reshape(
      value_layer,
      [batch_size, to_seq_length, num_attention_heads, size_per_head])

  # `value_layer` = [B, N, T, H]
  value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

  # `context_layer` = [B, N, F, H]
  context_layer = tf.matmul(attention_probs, value_layer)

  # `context_layer` = [B, F, N, H]
  context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

  if do_return_2d_tensor:
    # `context_layer` = [B*F, N*V]
    context_layer = tf.reshape(
        context_layer,
        [batch_size * from_seq_length, num_attention_heads * size_per_head])
  else:
    # `context_layer` = [B, F, N*V]
    context_layer = tf.reshape(
        context_layer,
        [batch_size, from_seq_length, num_attention_heads * size_per_head])

  return context_layer

# class lstm_layers(object):
#     def __init__(self,rnn_size,dropout_rate,num_layers):
#         self.rnn_size = rnn_size
#         self.dropout_rate = dropout_rate
#         pass
#
#     def blstm_layer(self,input):
#         """
#
#         :return:
#         """
#         with tf.variable_scope('rnn_layer'):
#             cell_fw = [self.get_rnn_cell() for _ in range(self.num_layers)]
#             cell_bw = [self.get_rnn_cell() for _ in range(self.num_layers)]
#             # if self.num_layers > 1:
#             #     cell_fw = rnn.MultiRNNCell([cell_fw] * self.num_layers, state_is_tuple=True)
#             #     cell_bw = rnn.MultiRNNCell([cell_bw] * self.num_layers, state_is_tuple=True)
#
#             rnn_output, _,_ = \
#                 stack_bidirectional_dynamic_rnn(cell_fw, cell_bw, input,
#                                                 sequence_length=self.lengths, dtype=tf.float32)
#             outputs = tf.concat(rnn_output, axis=2)
#         return outputs
