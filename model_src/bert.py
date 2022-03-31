from transformers import TFBertModel
import tensorflow as tf

class BertEmbeddings(tf.keras.layers.Layer):
    def __init__(self, **kwarg):
        super(BertEmbeddings, self).__init__(**kwarg)
        self.model = TFBertModel.from_pretrained("bert-base-cased")
        self.model.training = 'False'
        self.model.summary()

    def call(self, inputs, training=None, mask=None, **kwargs): 
        output = self.model(inputs)
        return output['last_hidden_state']

    def get_config(self):
        config = super(BertEmbeddings, self).get_config()
        config.update({'bert_config', self.model.config})
        return config