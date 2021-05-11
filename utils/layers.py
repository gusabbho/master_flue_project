import tensorflow as tf
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.layers import Layer, Conv1D, LayerNormalization, Add, Concatenate, LeakyReLU, Softmax,Dropout
import tensorflow_probability as tfp

POWER_ITERATIONS = 5

class Spectral_Norm_old(Constraint):
    '''
    Uses power iteration method to calculate a fast approximation 
    of the spectral norm (Golub & Van der Vorst)
    The weights are then scaled by the inverse of the spectral norm
    '''
    def __init__(self, power_iters=POWER_ITERATIONS):
        self.n_iters = power_iters

    def __call__(self, w):
        flattened_w = tf.reshape(w, [w.shape[0], -1])
        u = tf.random.normal([flattened_w.shape[0]])
        v = tf.random.normal([flattened_w.shape[1]])
        for i in range(self.n_iters):
            v = tf.linalg.matvec(tf.transpose(flattened_w), u)
            v = tf.keras.backend.l2_normalize(v)
            u = tf.linalg.matvec(flattened_w, v)
            u = tf.keras.backend.l2_normalize(u)
            sigma = tf.tensordot(u, tf.linalg.matvec(flattened_w, v), axes=1)
        return w / sigma

    def get_config(self):
        return {'n_iters': self.n_iters}
    
class Spectral_Norm(Constraint):
    '''
    Uses power iteration method to calculate a fast approximation 
    of the spectral norm (Golub & Van der Vorst)
    The weights are then scaled by the inverse of the spectral norm
    '''
    def __init__(self, power_iters=POWER_ITERATIONS):
        self.n_iters = power_iters
    
    def l2normalize(self, v, eps=1e-12):
        """l2 normalize the input vector.
        Args:
          v: tensor to be normalized
          eps:  epsilon (Default value = 1e-12)
        Returns:
          A normalized tensor
        """
        return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

    def __call__(self, w):
        w_shape = w.shape.as_list()
        w_mat = tf.reshape(w, [-1, w_shape[-1]]) 

        u = tf.random.normal([1, w_shape[-1]])

        for i in range(self.n_iters):
            v = self.l2normalize(tf.matmul(u, w_mat, transpose_b=True))
            u = self.l2normalize(tf.matmul(v, w_mat))

        sigma = tf.squeeze(tf.matmul(tf.matmul(v, w_mat), u, transpose_b=True))
        return w / sigma

    def get_config(self):
        return {'n_iters': self.n_iters}
    
class GumbelSoftmax_old(Layer):
    """
    Activation layer produces an approximate one hot encoded sample of the softmax for a which gradients are not zero as in the
    argmax function
    
    __init__()
    
    temperature: variable set to how agressiv the approximation of onehot vector should be
    
    call()
    
    logits: the linear output from your model 
    """
    def __init__(self,temperature = 0.5, *args, **kwargs):
        super(GumbelSoftmax,self).__init__()
        
        # Temperature
        self.tau = temperature
    
    def call(self, logits):
        U = tf.random.uniform(tf.shape(logits), minval=0, maxval=1, dtype=tf.dtypes.float32)
        g = -tf.math.log(-tf.math.log(U+1e-20)+1e-20)
        nom = tf.keras.activations.softmax((g + logits)/self.tau, axis=-1)
        return nom
    
class GumbelSoftmax(Layer):
    def __init__(self,temperature = 0.5, *args, **kwargs):
        super(GumbelSoftmax,self).__init__()
        
        # Temperature
        self.tau = temperature
        self.smx = Softmax()
    
    def call(self, logits):
        U = tf.random.uniform(tf.shape(logits), minval=0, maxval=1, dtype=tf.dtypes.float32)
        g = -tf.math.log(-tf.math.log(U+1e-20)+1e-20)
        prob = self.smx(logits)
        log_prob = tf.math.log(prob)
        nom = tf.keras.activations.softmax((g + logits)/self.tau, axis=-1)
        return nom
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'tau': self.tau
        })
        return config
    
class SelfAttention_old(Layer):
    def __init__(self, filters):
        super(SelfAttention, self).__init__()
        self.kernel_querry = Conv1D(max(1,filters//8), 1, padding= 'same', use_bias = False)
        self.kernel_key    = Conv1D(max(1,filters//8), 1, padding= 'same', use_bias = False)
        self.kernel_value  = Conv1D(max(1,filters//8), 1, padding= 'same', use_bias = False)
        self.out           = Conv1D(filters,    1, padding = 'same', use_bias = False)
        self.gamma = self.add_weight(name='gamma', initializer=tf.keras.initializers.Constant(value=1), trainable=True)
            
            
    def call(self, x, mask=None):
        querry = self.kernel_querry(x)
        key = self.kernel_key(x)
        value = self.kernel_value(x)
        attention_weights = tf.math.softmax(tf.matmul(querry, key, transpose_b = True), axis=1)
        attention_feature_map = tf.matmul(value, attention_weights, transpose_a = True)
        if mask is not None:
            attention_feature_map = tf.math.multiply(attention_feature_map, mask)
        attention_feature_map = tf.transpose(attention_feature_map, [0,2,1])
        
        out = x + self.out(attention_feature_map)*self.gamma
        
        return out, attention_weights
    
class SelfAttention(Layer):
    def __init__(self,  filters ):
        super(SelfAttention, self).__init__()
        
        self.kernel_querry = tf.keras.layers.Dense(filters)
        self.kernel_key    = tf.keras.layers.Dense(filters)
        self.kernel_value  = tf.keras.layers.Dense(filters)
        self.out           = tf.keras.layers.Dense(filters)
        self.num_heads = 8
        self.filters = filters
        self.depth = filters // self.num_heads
        self.gamma = self.add_weight(name='gamma', initializer=tf.keras.initializers.Constant(value=1), trainable=True)
        self.dout = Dropout(0.3)
        self.norm = LayerNormalization(axis = -1, epsilon = 1e-6)
        
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])        
    
    def call(self, x, mask=None, training = True):
        batch_size = tf.shape(x)[0]
        
        querry = self.kernel_querry(x)
        key    = self.kernel_key(x)
        value  = self.kernel_value(x)
        
        querry = self.split_heads( querry, batch_size)
        key    = self.split_heads( key, batch_size)
        value  = self.split_heads( value, batch_size)
         
        
        attention_logits  = tf.matmul(querry, key, transpose_b = True)
        attention_weights = tf.math.softmax(attention_logits, axis=-1)
        
        attention_feature_map = tf.matmul(attention_weights, value)
        if mask is not None:
            attention_feature_map = tf.math.multiply(attention_feature_map, mask)
            
        attention_feature_map = tf.transpose(attention_feature_map, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(attention_feature_map, (batch_size, -1, self.filters))
        concat_attention = self.dout(concat_attention, training = training)
        out = x + self.out(concat_attention)*self.gamma
        out = self.norm(out)
        return out, attention_weights

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters     
        })
        return config
    
class ResMod(Layer):
    """
    Residual module 
    """
    def __init__(self, filters, size, strides=1, dilation=1, constrains = None):
        super(ResMod, self).__init__()
        
        self.conv1 = Conv1D(filters, size,
                            dilation_rate = dilation,
                            padding = 'same',
                            use_bias = False,
                            kernel_constraint = constrains )
        self.conv2 = Conv1D(filters, size, dilation_rate = dilation, padding = 'same',
                            use_bias = False,
                            kernel_constraint = constrains )
        self.conv3 = Conv1D(filters, size, dilation_rate = dilation, padding = 'same',
                            use_bias = False,
                            kernel_constraint = constrains )
        self.strides = False
        
        if strides > 1:
            self.strides = True
            self.conv4 = Conv1D(filters, size, dilation_rate = 1, strides = strides, padding = 'same',
                            use_bias = False,
                            kernel_constraint = constrains )
            
            
        self.conv  = Conv1D(filters, 1, padding = 'same',
                            use_bias = False,
                            kernel_constraint = constrains )   
        self.add = Add()
        

        #self.norm = InstanceNormalization(
        #                           beta_initializer="random_uniform",
        #                           gamma_initializer="random_uniform")
        self.act = LeakyReLU(0.2)
        
    def call(self, x):
        x_in = self.conv(x)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.add([x, x_in])
        if self.strides:
            x = self.act(self.conv4(x)) 
        x = self.act(x)
        return x
