import os, sys
currentdir = os.path.dirname(os.getcwd())
sys.path.append(currentdir)

import argparse
import yaml
import datetime
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt

from utils.preprocessing import prepare_dataset
from utils import models_new
from utils import callbacks
from utils import losses


parser = argparse.ArgumentParser(""" """)

parser.add_argument('-c', '--config', type=str, default = 'config.yaml',
                   help = 'Configuration file that configures all parameters')

parser.add_argument('-v', '--verbose', action="store_true",
                   help = "Verbosity")
parser.add_argument('-g', '--gpu', type=str, choices=['0', '1','0,1'], default='0,1')

args = parser.parse_args()

def load_data(config):
    """ Function to load all the data """
    # Parameters
    file_thermo = config['file_thermo']
    file_meso   = config['file_meso']
    seq_length  = config['seq_length']
    max_samples = config['max_samples']
    
    thermo_train, thermo_val, n_thermo_train, n_thermo_val = prepare_dataset(file_thermo, 
                                                                             seq_length = seq_length,
                                                                             max_samples = max_samples)
    
    meso_train, meso_val, n_meso_train, n_meso_val = prepare_dataset(file_meso,
                                                                     seq_length = seq_length,
                                                                     max_samples = max_samples)

    data = {'thermo_train': thermo_train,
            'meso_train': meso_train,
            'thermo_val': thermo_val,
            'meso_val': meso_val,
            'n_thermo_train': n_thermo_train,
            'n_meso_train': n_meso_train,
            'n_thermo_val': n_thermo_val,
            'n_meso_val': n_meso_val}
    
    return data

def load_models(config):
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

    G    = models_new.Generator_res(G_filters, G_sizes, G_dilation, vocab, use_gumbel = G_gumbel, temperature = G_temperature)
    F    = models_new.Generator_res(G_filters, G_sizes, G_dilation, vocab, use_gumbel = G_gumbel, temperature = G_temperature) 
    D_x  = models_new.Discriminator(D_filters, D_sizes, D_strides, D_dilation, vocab, activation = D_activation)
    D_y  = models_new.Discriminator(D_filters, D_sizes, D_strides, D_dilation, vocab, activation = D_activation)
    
    return G, F, D_x, D_y

def load_classifier(config):
    vocab         = config['Vocab_size']
    filters       = config['filters']
    kernels       = config['kernels']
    dilations     = config['dilations']
    strides       = config['strides']
    use_attention = config['use_attention']
    file          = config['file']
    
    reg_model = models_new.Classifier(filters, kernels, strides, dilations, vocab)
    reg_model.load_weights(file)
    return reg_model

def load_losses(config):
    if config['loss'] == 'Non-Reduceing':
        loss_obj = losses.NonReduceingLoss()
    elif config['loss'] == 'Wasserstein':
        loss_obj = losses.WassersteinLoss()
    elif config['loss'] == 'Hinge':
        loss_obj = losses.HingeLoss()
    else:
        loss_obj = losses.NonReduceingLoss()
    return loss_obj

def load_optimizers(config):
    lr_D   = config['learning_rate_discriminator']
    lr_G   = config['learning_rate_generator']
    beta_D = config['beta_1_discriminator']
    beta_G = config['beta_1_generator']
    optimizers = {}
    if config['optimizer_discriminator'] == 'Adam':
        optimizers['opt_D_x'] = keras.optimizers.Adam(learning_rate = lr_D, beta_1 = beta_D) 
        optimizers['opt_D_y'] = keras.optimizers.Adam(learning_rate = lr_D, beta_1 = beta_D)
    else:
        optimizers['opt_D_x'] = keras.optimizers.SGD(learning_rate = lr_D, momentum = beta_D) 
        optimizers['opt_D_y'] = keras.optimizers.SGD(learning_rate = lr_D, momentum = beta_D)
        
    if config['optimizer_generator'] == 'Adam':
        optimizers['opt_G'] = keras.optimizers.Adam(learning_rate = lr_G, beta_1 = beta_G) 
        optimizers['opt_F'] = keras.optimizers.Adam(learning_rate = lr_G, beta_1 = beta_G)
    else: 
        optimizers['opt_G'] = keras.optimizers.SGD(learning_rate = lr_G, momentum = beta_G) 
        optimizers['opt_F'] = keras.optimizers.SGD(learning_rate = lr_G, momentum = beta_G)
        
    return optimizers

def load_metrics(config):
    metrics = {}
    metrics['loss_G']       = tf.keras.metrics.Mean('loss_G', dtype=tf.float32)
    metrics['loss_cycle_x'] = tf.keras.metrics.Mean('loss_cycle_x', dtype=tf.float32)
    metrics['loss_disc_y']  = tf.keras.metrics.Mean('loss_disc_y', dtype=tf.float32)
    metrics['loss_F']       = tf.keras.metrics.Mean('loss_F', dtype=tf.float32)
    metrics['loss_cycle_y'] = tf.keras.metrics.Mean('loss_cycle_y', dtype=tf.float32)
    metrics['loss_disc_x']  = tf.keras.metrics.Mean('loss_disc_x', dtype=tf.float32)

    metrics['temp_diff_x']  = tf.keras.metrics.Mean('temp_diff_x', dtype=tf.float32)
    metrics['temp_diff_y']  = tf.keras.metrics.Mean('temp_diff_y', dtype=tf.float32)

    metrics['acc_x']        = tf.keras.metrics.CategoricalAccuracy()
    metrics['cycled_acc_x'] = tf.keras.metrics.CategoricalAccuracy()
    metrics['acc_y']        = tf.keras.metrics.CategoricalAccuracy()
    metrics['cycled_acc_y'] = tf.keras.metrics.CategoricalAccuracy()
    
    return metrics

class VirusGan(tf.keras.Model):

    def __init__(self, config, callbacks=None):
        super(CycleGan, self).__init__()
        self.Generator, self.Discriminator = load_models(config['VirusGan'])
         
        self.add  = tf.keras.layers.Add()

    def compile( self, loss_obj, optimizers):
        
        super(CycleGan, self).compile()
        
        self.gen_optimizer = optimizers['Generator']
        self.disc_optimizer = optimizers['Discriminator']
        
        self.generator_loss_fn = loss_obj.generator_loss_fn
        self.discriminator_loss_fn = loss_obj.discriminator_loss_fn

    
    @tf.function
    def train_step(self, batch_data):

        

        with tf.GradientTape(persistent=True) as tape:
            
            # Batch data
            parents, child, W = batch_data
            
            # Generator output
            fake_child, _ = self.Generator(parent, training=True)

            # Discriminator output
            disc_real_child, _ = self.Discriminator(child, training=True)
            disc_fake_child, _ = self.Discriminator(fake_child, training=True)
            
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

def train(config, model, data, time):
    
    #file writers

    log_dir = os.path.join(config['Log']['base_dir'],time)
    
    Generator_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir,'Generator'))
    Discriminator_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir,'Discriminator'))



    # TODO: implement metrics
    metrics = load_metrics(config['CycleGan']['Metrics'])
    
    # Declare history object # TODO: Implement validation metric
    history = {
        "Generator_loss": [],
        "Discriminator_loss": []
        
    }

    for epoch in range(config['CycleGan']['epochs']):
        batches_x = data['meso_train'].shuffle(buffer_size = 40000).batch(config['CycleGan']['batch_size'], drop_remainder=True) 
        batches_y = data['thermo_train'].shuffle(buffer_size = 40000).batch(config['CycleGan']['batch_size'], drop_remainder=True)
        
        #Anneal schedule for gumbel
        if config['CycleGan']['Generator']['use_gumbel']:
                model.G.gms.tau = max(0.1, np.exp(-0.01*epoch))
                model.G.gms.tau = max(0.1, np.exp(-0.01*epoch))
                
        for step, x in enumerate(zip(batches_x,batches_y)):
            
            
            

            losses_, logits = model.train_step( batch_data = x)

            metrics['loss_G'](losses_["Gen_G_loss"]) 
            metrics['loss_cycle_x'](losses_["Cycle_X_loss"])
            metrics['loss_disc_y'](losses_["Disc_X_loss"])
            metrics['loss_F'](losses_["Gen_F_loss"]) 
            metrics['loss_cycle_y'](losses_["Cycle_Y_loss"])
            metrics['loss_disc_x'](losses_["Disc_Y_loss"])

            metrics['acc_x'](x[0][1], logits[0][0], x[0][2])
            metrics['acc_y'](x[1][1], logits[0][1], x[1][2])
            metrics['cycled_acc_x'](x[0][1], logits[1][0], x[0][2])
            metrics['cycled_acc_y'](x[1][1], logits[1][1], x[1][2])
        
        
        diff_x=0
        diff_y=0
        if epoch % 10 == 0:
            val_x = data['meso_val'].shuffle(buffer_size = 40000).batch(1, drop_remainder=False)
            val_y = data['thermo_val'].shuffle(buffer_size = 40000).batch(1, drop_remainder=False)
            
            diff_x, diff_y = model.validate_step( val_x, val_y,data, epoch)

            with temp_diff_summary_x.as_default():
                tf.summary.scalar('temp_diff', diff_x, step=epoch, description = 'temp_diff_x')
            with temp_diff_summary_y.as_default():
                tf.summary.scalar('temp_diff', diff_y, step=epoch, description = 'temp_diff_y')


        if args.verbose:    
            print("Epoch: %d Loss_G: %2.4f Loss_F: %2.4f Loss_cycle_X: %2.4f Loss_cycle_Y: %2.4f Loss_D_Y: %2.4f Loss_D_X %2.4f" % 
              (epoch, float(metrics['loss_G'].result()),
               float(metrics['loss_F'].result()),
               float(metrics['loss_cycle_x'].result()),
               float(metrics['loss_cycle_y'].result()),
               float(metrics['loss_disc_y'].result()),
               float(metrics['loss_disc_x'].result())))
            print("Epoch: %d acc trans x: %2.4f acc trans y: %2.4f acc cycled x : %2.4f acc cycled y: %2.4f" % 
              (epoch, metrics['acc_x'].result(),
               metrics['acc_y'].result(),
               metrics['cycled_acc_x'].result(),
               metrics['cycled_acc_y'].result()))

        # Write log file
        with G_summary_writer.as_default():
                tf.summary.scalar('loss', metrics['loss_G'].result(), step = epoch, description = 'X transform')
                tf.summary.scalar('acc', metrics['acc_x'].result(), step = epoch, description = 'X transform' )


        with F_summary_writer.as_default():
            tf.summary.scalar('loss', metrics['loss_F'].result(), step = epoch, description = 'Y transform')
            tf.summary.scalar('acc', metrics['acc_y'].result(), step = epoch, description = 'Y transform' )

        with D_x_summary_writer.as_default():         
            tf.summary.scalar('loss', metrics['loss_disc_y'].result(), step = epoch, description = 'X discriminator')        
        with D_y_summary_writer.as_default():        
            tf.summary.scalar('loss', metrics['loss_disc_x'].result(), step = epoch, description = 'Y discriminator')    
        with X_c_summary_writer.as_default(): 
            tf.summary.scalar('loss', metrics['loss_cycle_x'].result(), step = epoch, description = 'X cycle')
            tf.summary.scalar('acc', metrics['cycled_acc_x'].result(), step = epoch, description = 'X cycle' )         
        with Y_c_summary_writer.as_default():
            tf.summary.scalar('loss', metrics['loss_cycle_y'].result(), step = epoch, description = 'Y cycle')
            tf.summary.scalar('acc', metrics['cycled_acc_y'].result(), step = epoch, description = 'Y cycle' )

        # Save history object
        history["Gen_G_loss"].append(metrics['loss_G'].result().numpy())
        history["Cycle_X_loss"].append(metrics['loss_cycle_x'].result().numpy())
        history["Disc_X_loss"].append(metrics['loss_disc_x'].result().numpy())
        history["Gen_F_loss"].append(metrics['loss_F'].result().numpy())
        history["Cycle_Y_loss"].append(metrics['loss_cycle_y'].result().numpy())
        history["Disc_Y_loss"].append(metrics['loss_disc_y'].result().numpy())
        history["x_acc"].append(metrics['acc_x'].result().numpy())
        history["x_c_acc"].append(metrics['cycled_acc_x'].result().numpy())
        history["y_acc"].append(metrics['acc_y'].result().numpy())
        history["y_c_acc"].append(metrics['cycled_acc_y'].result().numpy())
        history["temp_diff_x"].append(diff_x.numpy())
        history["temp_diff_y"].append(diff_y.numpy())
        # Reset states
        metrics['loss_G'].reset_states()
        metrics['loss_cycle_x'].reset_states()
        metrics['loss_disc_y'].reset_states()
        metrics['loss_F'].reset_states() 
        metrics['loss_cycle_y'].reset_states()
        metrics['loss_disc_x'].reset_states()

        metrics['acc_x'].reset_states()
        metrics['acc_y'].reset_states()
        metrics['cycled_acc_x'].reset_states()
        metrics['cycled_acc_y'].reset_states()
    
    return history


def main():
    
    # GPU setting

    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
    
    # Get time
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # Load configuration file
    with open(args.config, 'r') as file_descriptor:
        config = yaml.load(file_descriptor, Loader=yaml.FullLoader)
        
    with open(args.config, 'r') as file_descriptor:
        config_str = file_descriptor.read()

    # Load training data
    data = load_data(config['Data'])
    
    # Callbacks
    cb = callbacks.PCAPlot(data['thermo_train'].as_numpy_iterator(), data['meso_train'].as_numpy_iterator(), data['n_thermo_train'], data['n_meso_train'], logdir=os.path.join(config['Log']['base_dir'],time,'img')) 
    
    # Initiate model
    model = CycleGan(config, callbacks = cb)
    
    loss_obj  = load_losses(config['CycleGan']['Losses'])
    optimizers = load_optimizers(config['CycleGan']['Optimizers'])
    model.compile(loss_obj, optimizers)
    
    # Initiate Training

    history = train(config, model, data, time)
    
    #writing results
    
    result_dir = os.path.join(config['Results']['base_dir'],time)
    os.mkdir(os.path.join(result_dir))
    os.mkdir(os.path.join(result_dir,'weights'))
    # Save model
    model.save_weights(os.path.join(result_dir,'weights','cycle_gan_model'))
    # Write history obj
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(result_dir,'history.csv'))
    # Save config_file
    with open(os.path.join(result_dir, 'config.yaml'), 'w') as file_descriptor:
        file_descriptor.write(config_str)
        
    return 0





# Training

    
if __name__ == "__main__":
    main()