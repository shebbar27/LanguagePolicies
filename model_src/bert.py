from transformers import BertTokenizer, TFBertModel
import tensorflow as tf

class BertEmbeddings(tf.keras.layers.Layer):
    def __init__(self, **kwarg):
        super(BertEmbeddings, self).__init__(**kwarg)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        self.model = TFBertModel.from_pretrained("bert-base-cased")
    
    def call(self, inputs, training=None, mask=None, **kwargs): 
        # inputs = self.tokenizer(inputs, return_tensors="tf")
        output = self.model(inputs)
        return output['pooler_output']
