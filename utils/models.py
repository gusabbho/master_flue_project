import tensorflow as tf
#import tensorflow_addons as tfa
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D, Dense, Flatten, LeakyReLU, Embedding, Input, LeakyReLU, LayerNormalization, BatchNormalization, Softmax
from utils.layers_new import SelfAttention, ResMod, Spectral_Norm, GumbelSoftmax 

class Generator_res(Model):
    def __init__(self, filters, size, dilation, vocab, use_gumbel, temperature = 0.5):
        super(Generator_res, self).__init__()
        self.res1 = ResMod(filters[0], size[0], dilation = dilation[0])
        self.res2 = ResMod(filters[1], size[1], dilation = dilation[1])
        self.res3 = ResMod(filters[2], size[2], dilation = dilation[2])
        self.res4 = ResMod(filters[3], size[3], dilation = dilation[3])
        self.res5 = ResMod(filters[4], size[4], dilation = dilation[4])
        self.res6 = ResMod(filters[5], size[5], dilation = dilation[5])
        
        self.atte = SelfAttention(256)
        if use_gumbel:
            self.gms = GumbelSoftmax(temperature = 0.5)
        else:
            self.gms = Softmax()
        self.out = Conv1D(vocab, 3, padding = 'same', activation = self.gms)
    def call(self, x):
        
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x, a_w = self.atte(x)
        x = self.res6(x)
        x = self.out(x)
        return x, a_w
    
class Discriminator(Model):
    def __init__(self, filters, size, strides, dilation, vocab, activation = 'sigmoid' ):
        super(Discriminator, self).__init__()
        self.constraint = Spectral_Norm()
        self.act = LeakyReLU(0.2)
        self.conv1 = Conv1D(filters[0], size[0], strides=strides[0], padding='same', kernel_constraint = self.constraint, use_bias = False)
        self.conv2 = Conv1D(filters[0], size[0], strides=strides[0], padding='same', kernel_constraint = self.constraint, use_bias = False)
        self.conv3 = Conv1D(filters[0], size[0], strides=strides[0], padding='same', kernel_constraint = self.constraint, use_bias = False)
        self.conv4 = Conv1D(filters[0], size[0], strides=strides[0], padding='same', kernel_constraint = self.constraint, use_bias = False)
        self.atte = SelfAttention(vocab)
        self.conv = Conv1D(1, 4, strides=1, activation= activation, padding='same', kernel_constraint = self.constraint, use_bias = False)
        
    def call(self, x):
        x, a_w = self.atte(x)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        return self.conv(x), a_w