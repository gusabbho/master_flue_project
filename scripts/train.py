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
from utils import models
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
    file_parents = config['file_parents']
    file_children   = config['file_children']
    seq_length  = config['seq_length']
    max_samples = config['max_samples']
    
    train, test = prepare_dataset("../../../../parent_sequences_translated.fasta", file_children, seq_length = seq_length, max_samples = max_samples)
    

    data = {"Train-Data": train,
           "Test-Data": test}
    
    return data

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
        optimizers['Discriminator'] = keras.optimizers.Adam(learning_rate = lr_D, beta_1 = beta_D) 
    else:
        optimizers['Discriminator'] = keras.optimizers.SGD(learning_rate = lr_D, momentum = beta_D) 

    if config['optimizer_generator'] == 'Adam':
        optimizers['Generator'] = keras.optimizers.Adam(learning_rate = lr_G, beta_1 = beta_G) 
    else: 
        optimizers['Generator'] = keras.optimizers.SGD(learning_rate = lr_G, momentum = beta_G) 
        
    return optimizers

# TODO: validation metric
def load_metrics(config):
    metrics = {}
    metrics['Generator_loss']      = tf.keras.metrics.Mean('Generator_loss', dtype=tf.float32)
    metrics['Discriminator_loss']  = tf.keras.metrics.Mean('Discriminator_loss', dtype=tf.float32)
    return metrics



def train(config, model, data, time):
    
    #file writers

    log_dir = os.path.join(config['Log']['base_dir'],time)
    
    Generator_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir,'Generator'))
    Discriminator_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir,'Discriminator'))



    # TODO: implement metrics for validation
    metrics = load_metrics(config['VirusGan']['Metrics'])
    
    # Declare history object # TODO: Implement validation metric
    history = {
        "Generator_loss": [],
        "Discriminator_loss": []    
    }

    # 
    for epoch in range(config['VirusGan']['epochs']):
        # TODO change buffer size to fit data set 
        batches = data['Train-Data'].shuffle(buffer_size = 40000).batch(config['VirusGan']['batch_size'], drop_remainder=True) 
  
        #Anneal schedule for gumbel
        if config['VirusGan']['Generator']['use_gumbel']:
                model.Generator.gms.tau = max(0.3, np.exp(-0.01*epoch))
                
        for step, x in enumerate(batches):
            
            loss, logits = model.train_step(batch_data = x)

            metrics['Generator_loss'](loss["Generator_loss"]) 
            metrics['Discriminator_loss'](loss["Discriminator_loss"])

        # TODO Validation of training
        if epoch % 10 == 0:
            pass



        if args.verbose:    
            print("Epoch: %d Generator loss: %2.4f  Discriminator loss: %2.4f" % 
              (epoch, float(metrics['Generator_loss'].result()),
               float(metrics['Discriminator_loss'].result())))
            # TODO validation
            #print("Epoch: %d acc trans x: %2.4f acc trans y: %2.4f acc cycled x : %2.4f acc cycled y: %2.4f" % 
             # (epoch, metrics['val_***'].result(),
             #  metrics['val_***'].result()))
               

        # Write log file
        with Generator_summary_writer.as_default():
            tf.summary.scalar('loss', metrics['Generator_loss'].result(), step = epoch, description = 'Generator Loss')

        with Discriminator_summary_writer.as_default():         
            tf.summary.scalar('loss', metrics['Discriminator_loss'].result(), step = epoch, description = 'Discriminator Loss')        


        # Save history object
        history["Generator_loss"].append(metrics['Generator_loss'].result().numpy())
        history["Discriminator_loss"].append(metrics['Discriminator_loss'].result().numpy())
        
        # Reset states
        metrics['Generator_loss'].reset_states()
        metrics['Discriminator_loss'].reset_states()
        
    
    return history


def main():
    
    # GPU setting

    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
    
    # Get time stamp
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Load configuration file
    with open(args.config, 'r') as file_descriptor:
        config = yaml.load(file_descriptor, Loader=yaml.FullLoader)
        
    with open(args.config, 'r') as file_descriptor:
        config_str = file_descriptor.read()

    # Load training data
    data = load_data(config['Data'])
    
    # Initiate model
    model = models.VirusGan(config)
    # Compile model
    loss_obj  = load_losses(config['VirusGan']['Losses'])
    optimizers = load_optimizers(config['VirusGan']['Optimizers'])
    model.compile(loss_obj, optimizers)
    
    # Initiate Training
    history = train(config, model, data, time)
    
    # check results dir
    if not os.path.isdir(config['Results']['base_dir']):
        os.mkdir(config['Results']['base_dir'])
    #writing results   
    result_dir = os.path.join(config['Results']['base_dir'],time)
    os.mkdir(os.path.join(result_dir))
    os.mkdir(os.path.join(result_dir,'weights'))
    
    # Save model
    model.save_weights(os.path.join(result_dir,'weights','virus_gan_model'))
    
    # Write history obj
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(result_dir,'history.csv'))
    
    # Save config_file
    with open(os.path.join(result_dir, 'config.yaml'), 'w') as file_descriptor:
        file_descriptor.write(config_str)
        
    return 0

    
if __name__ == "__main__":
    main()