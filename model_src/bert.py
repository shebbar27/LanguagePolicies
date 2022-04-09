from transformers import TFBertModel
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()

class BertEmbeddings(tf.keras.layers.Layer):
    def __init__(self, **kwarg):
        super(BertEmbeddings, self).__init__(**kwarg)
        self.model = TFBertModel.from_pretrained("bert-base-cased")
        self.model.training = 'False'
        self.model.summary()

    def call(self, inputs, training=None, mask=None, **kwargs): 
        output = self.model(inputs)
        output = output['pooler_output']
        outputShape = tf.shape(output)
        output = tf.reshape(output, [outputShape[0], 1, outputShape[1]])
        return output

    def get_config(self):
        config = super(BertEmbeddings, self).get_config()
        config.update({'bert_config', self.model.config})
        return config