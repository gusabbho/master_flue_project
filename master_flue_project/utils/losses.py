import tensorflow as tf
from tensorflow.keras.losses import Loss, CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.backend import softplus
from tensorflow.keras import backend as K

class WassersteinLoss(Loss):
    """
    Wasserstein loss 
        Generator loss function generator_loss_fn(fake) 
            fake output from discriminator
        Discriminator loss function discriminator_loss_fn(real, fake)
            real and fake output from discriminator
    """
    def __init__(self, ):
        super(WassersteinLoss, self).__init__()
    
    # Define the loss function for the generators
    def generator_loss_fn(self, fake):
        return -tf.math.reduce_mean(fake, axis=0)

    # Define the loss function for the discriminators
    def discriminator_loss_fn(self, real, fake):
        
        real_loss = tf.math.reduce_mean(real)
        fake_loss = tf.math.reduce_mean(fake)
        return fake_loss - real_loss
    
class NonReduceingLoss(Loss):
    """
    Non-Reduceing loss 
        Generator loss function generator_loss_fn(fake) 
            fake output from discriminator
        Discriminator loss function discriminator_loss_fn(real, fake)
            real and fake output from discriminator
    """
    def __init__(self, ):
        super(NonReduceingLoss, self).__init__()
        
    def generator_loss_fn(self, fake):
        return -tf.math.reduce_mean(tf.math.log(fake))
        #return K.mean(K.softplus(-fake), axis=0)
    
    def discriminator_loss_fn(self, real, fake):
        L1 = tf.math.reduce_mean(tf.math.log(real))
        L2 = tf.math.reduce_mean(tf.math.log(tf.ones_like(fake)-fake))
        #L1 = K.mean(K.softplus(-real), axis=0)
        #L2 = K.mean(K.softplus(fake), axis=0)
        return -1*(L1 + L2)
    
class HingeLoss(Loss):
    """
    Hinge loss 
        Generator loss function generator_loss_fn(fake) 
            fake output from discriminator
        Discriminator loss function discriminator_loss_fn(real, fake)
            real and fake output from discriminator
    """
    def __init__(self ):
        super(HingeLoss,self).__init__()
    
    # Define the loss function for the generators
    def generator_loss_fn(self, fake):
        return -1 * K.mean(fake, axis=0)

    # Define the loss function for the discriminators
    def discriminator_loss_fn(self, real, fake):
        loss = K.mean(K.relu(1. - real),axis=0)
        loss += K.mean(K.relu(1. + fake),axis=0)
        return loss