from keras_dna import Generator, ModelWrapper
from keras_dna.model import load_wrapper
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import yaml

from MyModuleLibrary.mykeras.losses import mae_cor, correlate
import os
import pyBigWig
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

with open("./configuration.yml", 'r') as f:
    configuration = yaml.safe_load(f)
  
annotation_file = configuration['annotation_file']
file_to_store_model = configuration['file_to_store_model']
annotation_type = configuration['annotation_type']

if annotation_type == 'pol2':
  window = 2048
  tg_window = 16

elif annotation_type == 'cohesine':
  window = 32768
  tg_window = 256

inputs = tf.keras.layers.Input(shape=(2048, 4))

x = tf.keras.layers.Conv1D(32, 12, activation='relu', padding='same')(inputs)
x = tf.keras.layers.MaxPooling1D(8, padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.2)(x)

x = tf.keras.layers.Conv1D(32, 5, activation='relu', padding='same')(x)
x = tf.keras.layers.MaxPooling1D(4, padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.2)(x)

x = tf.keras.layers.Conv1D(32, 5, activation='relu', padding='same')(x)
x = tf.keras.layers.MaxPooling1D(4, padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.2)(x)

x = tf.keras.layers.Conv1D(16, 5, activation='relu', padding='same', dilation_rate=2)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.2)(x)

x1 = tf.keras.layers.Conv1D(16, 5, activation='relu', padding='same', dilation_rate=4)(x)
x1 = tf.keras.layers.BatchNormalization()(x1)
x1 = tf.keras.layers.Dropout(0.2)(x1)

x2 = tf.keras.layers.Concatenate(axis=-1)([x, x1])
x2 = tf.keras.layers.Conv1D(16, 5, activation='relu', padding='same', dilation_rate=8)(x2)
x2 = tf.keras.layers.BatchNormalization()(x2)
x2 = tf.keras.layers.Dropout(0.2)(x2)

x3 = tf.keras.layers.Concatenate(axis=-1)([x, x1, x2])
x3 = tf.keras.layers.Conv1D(16, 5, activation='relu', padding='same', dilation_rate=16)(x3)
x3 = tf.keras.layers.BatchNormalization()(x3)
x3 = tf.keras.layers.Dropout(0.2)(x3)

x = tf.keras.layers.Concatenate(axis=-1)([x, x1, x2, x3])
outputs = tf.keras.layers.Conv1D(1, 1, activation='linear', padding='same')(x)

model = tf.keras.models.Model(inputs, outputs)
model.compile(optimizer='adam',
              sample_weight_mode='temporal',
              loss='mae',
              metrics=[correlate])

generator_train = Generator(batch_size=64,
                            fasta_file='./data/saccer3.fa',
                            annotation_files=[annotation_file],
                            size='./data/saccer3.chrom.sizes',
                            window=window,
                            tg_window=tg_window,
                            downsampling='mean',
                            overlapping=10,
                            weighting_mode=([[0, 1, 2, 3, 4, 5, 6, 10]],
                                            [[1, 4, 10, 20, 30, 40, 50]]),
                            incl_chromosomes=['chr01', 'chr02', 'chr03', 'chr04',
                                              'chr05', 'chr06', 'chr07', 'chr08', 'chr09', 'chr10',
                                              'chr11', 'chr12', 'chr13', 'chr14'],
                            output_shape=(64, 16, 1))

wrap = ModelWrapper(model=model,
                    generator_train=generator_train,
                    validation_chr=['chr15', 'chr16'],
                    weights_val=True)

early = keras.callbacks.EarlyStopping(monitor='val_loss',
                                      min_delta=0,
                                      patience=5,
                                      verbose=0,
                                      mode='auto')

checkpointer = keras.callbacks.ModelCheckpoint(filepath=file_to_store_model,
                                               monitor='val_loss',
                                               verbose=0, 
                                               save_best_only=True, 
                                               save_weights_only=False, 
                                               mode='min',
                                               period=1)

history = wrap.train(epochs=100,
                     callbacks=[early, checkpointer],
                     verbose=1)
