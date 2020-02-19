import aux_fn
import numpy as np
import argparse
import yaml
import os
import pickle
from tqdm import tqdm
import hashlib
import logging
from time import time

NR_INPUTS = 117  # this is the number of samples that go to inferencing - for 30 sec long audio with 0.25s for rolling window and 1s fed to the network, used just for time measurements


def get_response(model, file, file_meta, sample_rate, rolling_step=None):
    window_width = model.layers[0].input_shape[1]
    if rolling_step is None:
        rolling_step = window_width // 2
    data = aux_fn.get_data_from_files(
        [file], sample_rate, window_width, rolling_step)
    print(data.shape)

    yhat = model.predict(np.expand_dims(data, axis=2),
                         batch_size=data.shape[0], verbose=1)
    yhat = np.squeeze(yhat)
    response = aux_fn.unroll(yhat, rolling_step)
    raw_data = aux_fn.unroll(data, rolling_step)

    anomaly_bounds = None
    if file_meta['event_present']:
        ev_start = round(
            file_meta['event_start_in_mixture_seconds'] * sample_rate)
        ev_end = round(
            ev_start + file_meta['event_length_seconds'] * sample_rate)
        anomaly_bounds = (min(ev_start, len(raw_data)),
                          min(ev_end, len(raw_data)))

    return {'data': raw_data, 'resp': response, 'actual': anomaly_bounds}


with open("train_parameters.yaml", 'r') as stream:
    param = yaml.load(stream, Loader=yaml.FullLoader)
with open("test_parameters.yaml", 'r') as stream:
    test_param = yaml.load(stream, Loader=yaml.FullLoader)


sample_rate = param['sample_rate']
LAYER_SIZE = param['LAYER_SIZE']  # amount of neurons in each hidden layer
HIDDEN_LAYER_COUNT = param['HIDDEN_LAYER_COUNT']  # amount of hidden layers
# this is size of window to put into autoencoder.
window_width = param['window_width']
rolling_step = test_param['rolling_step']  # this is rolling step
epochs = param['train_epochs']  # epochs

model_file = './trained_model/saved_model.h5'

test_folder = test_param['test_path']+"/audio"
yaml_file = test_param['test_path'] + \
    "/meta/mixture_recipes_devtest_gunshot.yaml"

print("Testing folder to proceed:")
print(test_folder)

m = hashlib.md5()
m.update(model_file.encode('utf-8'))
m.update(test_param['test_path'].encode('utf-8'))
response_hash = m.hexdigest()

HIDDEN_LAYER_COUNT = 8
model = aux_fn.gen_model(window_width, LAYER_SIZE, HIDDEN_LAYER_COUNT)
model.load_weights(model_file)
with open(yaml_file, 'r') as infile:
    metadata = yaml.load(infile, Loader=yaml.SafeLoader)

files = aux_fn.get_all_files(test_folder)
sig_process = {}

logging.basicConfig(filename='trained_model/inference.log',
                    filemode='w', level=logging.INFO)

for file in tqdm(files):
    file_time = time()
    print(file)
    logging.info(f' inference on {file}: ')

    filename_wav = os.path.basename(file)
    for i in range(len(metadata)):
        if filename_wav in metadata[i]['mixture_audio_filename']:
            file_meta = metadata[i]
            break
    sig_process[filename_wav] = get_response(
        model, file, file_meta, sample_rate, rolling_step)
    tmp_time = time()-file_time
    logging.info(
        f" \tfile inference time = {tmp_time:.2f} s, {(tmp_time/117)*1000:.4f} ms per sample")


response_file = 'trained_model/response_'+response_hash+'.pickle'

with open(response_file, 'wb') as f:
    pickle.dump(sig_process, f)
