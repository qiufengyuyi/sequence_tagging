import tensorflow as tf


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

