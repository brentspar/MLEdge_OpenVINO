from __future__ import absolute_import, division, print_function, unicode_literals
import librosa

from tqdm import tqdm
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
import numpy as np
import os

import aux_fn
import logging
import time

#parameters
sample_rate=16000
window_width=sample_rate #this is size of window to put into autoencoder.
LAYER_SIZE=15 # amount of neurons in each hidden layer
HIDDEN_LAYER_COUNT=8 #amount of hidden layers

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(filename="results/training.log",filemode='w', level=logging.INFO)

path = '../data/source_data/bgs/audio'

class LogHistory(Callback):
    def on_epoch_end(self, batch, logs={}):
        logging.info(logs)


files = aux_fn.get_all_files(path)
print(['Number of files',len(files)])
data = aux_fn.get_data_from_files(files, sr=sample_rate, ww=window_width)
data = np.expand_dims(data, axis=2)

model = aux_fn.gen_model(window_width, LAYER_SIZE, HIDDEN_LAYER_COUNT)

start_time=time.time()
model.fit(data, data, epochs=100, batch_size=300, verbose=1, shuffle=True,callbacks=[LogHistory()])
logging.info("training time=%f",time.time()-start_time)

model.save("results/new_model.h5")
