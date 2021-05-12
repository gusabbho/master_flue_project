import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D, Dense, Flatten, LeakyReLU, Embedding, Input, LeakyReLU, LayerNormalization, BatchNormalization, Softmax, Concatenate,Dropout
from utils.layers import SelfAttention, ResMod, Spectral_Norm, GumbelSoftmax
from utils import preprocessing as pre



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
        #TODO ADD in noise
        shape = tf.shape(x)
        noise= tf.random.normal(shape)
        #newe_data = our_data + noise
        x = self.res1(x + noise)
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
        self.conv2 = Conv1D(filters[1], size[0], strides=strides[0], padding='same', kernel_constraint = self.constraint, use_bias = False)
        self.conv3 = Conv1D(filters[2], size[0], strides=strides[0], padding='same', kernel_constraint = self.constraint, use_bias = False)
        self.conv4 = Conv1D(filters[3], size[0], strides=strides[0], padding='same', kernel_constraint = self.constraint, use_bias = False)
        self.atte = SelfAttention(vocab*2)
        self.conv = Conv1D(1, 4, strides=1, activation= activation, padding='same', kernel_constraint = self.constraint, use_bias = False)
        self.cat = Concatenate(axis=-1)
        
    def call(self, parent, child):
        x = self.cat([parent, child])
        x, a_w = self.atte(x)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        return self.conv(x), a_w
    
class VirusGan(tf.keras.Model):

    def __init__(self, config):
        super(VirusGan, self).__init__()
        self.Generator, self.Discriminator = self.load_models(config['VirusGan'])
         
        self.add  = tf.keras.layers.Add()

    def compile( self, loss_obj, optimizers):
        
        super(VirusGan, self).compile()
        
        self.gen_optimizer = optimizers['Generator']
        self.disc_optimizer = optimizers['Discriminator']
        
        self.generator_loss_fn = loss_obj.generator_loss_fn
        self.discriminator_loss_fn = loss_obj.discriminator_loss_fn

    def load_models(self, config):
        """Create all models that is used in cycle gan""" 

        model_type = config["Generator"]["type"]
        G_filters = config["Generator"]["filters"]
        G_sizes   = config["Generator"]["kernels"]
        G_dilation= config["Generator"]["dilations"]
        G_gumbel = config["Generator"]["use_gumbel"]
        G_temperature = config["Generator"]["temperature"]


        D_filters = config["Discriminator"]["filters"]
        D_sizes   = config["Discriminator"]["kernels"]
        D_dilation= config["Discriminator"]["dilations"]
        D_strides = config["Discriminator"]["strides"]

        if config["Losses"]["loss"] == 'Non-Reducing':
            D_activation = 'sigmoid'
        else:
            D_activation = 'linear'

        vocab = config["Vocab_size"] 

        generator = Generator_res(G_filters, G_sizes, G_dilation, vocab, use_gumbel = G_gumbel, temperature = G_temperature)
         
        discriminator = Discriminator(D_filters, D_sizes, D_strides, D_dilation, vocab, activation = D_activation)

        return generator, discriminator
    
    @tf.function
    def train_step(self, batch_data):

        

        with tf.GradientTape(persistent=True) as tape:
            
            # Batch data
            parent, child = batch_data
            
            # Generator output
            fake_child, _ = self.Generator(parent, training=True)

            # Discriminator output
            disc_real_child, _ = self.Discriminator(parent, child, training=True)
            disc_fake_child, _ = self.Discriminator(parent, fake_child, training=True)
            
            # Loss for generator and discriminator
            gen_loss = self.generator_loss_fn(disc_fake_child)
            disc_loss = self.discriminator_loss_fn(disc_real_child, disc_fake_child)

        # Get the gradients for the generator
        gen_grads = tape.gradient(gen_loss, self.Generator.trainable_variables)
       
        # Get the gradients for the discriminators
        disc_grads = tape.gradient(disc_loss, self.Discriminator.trainable_variables)


        # Update the weights of the generator 
        self.gen_optimizer.apply_gradients(zip(gen_grads, self.Generator.trainable_variables))  

        # Update the weights of the discriminator
        self.disc_optimizer.apply_gradients(zip(disc_grads, self.Discriminator.trainable_variables))


        return {
            "Generator_loss": gen_loss,
            "Discriminator_loss": disc_loss,
        }, fake_child
    
    #@tf.function
    def validate_step(self, batch_data):
        # TODO 
        return 0
    #@tf.function
    def generate(self, data):
        parents, children = data
        fake = []
        for parent in parents:
            shape = tf.shape(parents)
            parent = tf.reshape(parents, (1, shape[0], shape[1])):
            fake_child, _ = self.Generator(parent, training=False)
            fake.append(fake_child.numpy())
        seqs = []
        for seq in fake:
            seqs.append(pre.convert_table(seq))
        return seqs
        return 0
