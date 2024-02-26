# Custom L1 Distance layer mofulr

#Import dependencies
import tensorflow as tf
from tensorflow.keras.layers import Layer
#custom L1 Distance layer
class L1Dist(Layer):

    #Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()

    #Similarity claculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding -validation_embedding)
