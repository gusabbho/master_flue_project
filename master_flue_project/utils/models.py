import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D, Dense, Flatten, LeakyReLU, Embedding, Input, LeakyReLU, LayerNormalization, BatchNormalization, Softmax, Concatenate,Dropout
from utils.layers import SelfAttention, ResMod, Spectral_Norm, GumbelSoftmax
from utils import preprocessing as pre



class Generator_res(Model):
    def __init__(self, config, filters, size, dilation, vocab, use_gumbel, temperature = 0.5):
        super(Generator_res, self).__init__()
        self.n_l = config["layers"]
        self.res = []
        for i in range(self.n_l):
            self.res.append(ResMod(filters[i], size[i], dilation = dilation[i]))

        self.concat = Concatenate()
        self.atte_loc = config['attention_loc']
        self.atte = SelfAttention(filters[self.atte_loc])
        if use_gumbel:
            self.gms = GumbelSoftmax(temperature = 0.5)
        else:
            self.gms = Softmax()
        self.out = Conv1D(vocab, 3, padding = 'same', activation = self.gms)
    def call(self, x):
        shape = tf.shape(x)
        noise= tf.random.normal(shape)
        x = self.concat([x, noise])
        for i in range(self.n_l):
            x = self.res[i](x)
            if self.atte_loc == i:
                x, a_w = self.atte(x)
        x = self.out(x)
        return x, a_w

class Discriminator(Model):
    def __init__(self,config, filters, size, strides, dilation, vocab, activation = 'sigmoid' ):
        super(Discriminator, self).__init__()

        self.act = LeakyReLU(0.2)
        self.n_l = config["layers"]
        self.conv = []
        for i in range(self.n_l):
            self.conv.append(tfa.layers.SpectralNormalization(Conv1D(filters[i],
                                                                     size[i],
                                                                     strides=strides[i],
                                                                     padding='same',
                                                                     use_bias = False)))

        self.atte_loc = config["attention_loc"]
        self.atte = SelfAttention( filters[self.atte_loc])
        self.flat  = Flatten()
        self.dense = tfa.layers.SpectralNormalization(Dense(1,
                                                            activation= activation,
                                                            use_bias = False))
        self.cat = Concatenate(axis=-1)

    def call(self, parent, child):
        x = self.cat([parent, child])

        for i in range(self.n_l):
            x = self.act(self.conv[i](x))
            if self.atte_loc == i:
                x, a_w = self.atte(x)

        x = self.flat(x)
        return self.dense(x), a_w

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

        generator = Generator_res(config["Generator"],G_filters, G_sizes, G_dilation, vocab, use_gumbel = G_gumbel, temperature = G_temperature)

        discriminator = Discriminator(config["Discriminator"],D_filters, D_sizes, D_strides, D_dilation, vocab, activation = D_activation)

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
    def generate(self, data, n_children = 32, parent_ids = None) -> dict:
        """
        NOTE!! parent_ids are assumed to match the order of the encoded data

        Return a dictionary like {parent_id: [child_sequences (list)]}
        """
        fake = []
        for n_p, d in enumerate(data):
            parent, child = d
            shape = tf.shape(parent)
            parent = tf.reshape(parent, (1, shape[0], shape[1]))
            parent = tf.repeat(parent, repeats=n_children, axis=0)
            fake_child, _ = self.Generator(parent, training=False)
            fake_child = tf.math.argmax(fake_child, axis=-1)
            fake.append(fake_child.numpy())

        seqs = {}
        if parent_ids is None:
            fake_data = enumerate(fake)
        else:
            fake_data = zip(parent_ids, fake)

        for parent_id, parent in fake_data:
            seqs[parent_id] = []
            for seq in parent:
                seqs[parent_id].append(pre.convert_table(seq))
        return seqs

