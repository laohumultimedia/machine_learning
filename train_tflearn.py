from __future__ import division, print_function, absolute_import
import numpy as np
import tflearn
from os.path import isfile
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, upsample_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression


data = np.load("video.npy")
_,height, width, channel = data.shape
print(data.shape)
print('w:%i, h:%i' % (width, height))
data = data / 255.
print('[+] Building CNN')
network = input_data(shape = [None, height, width, 3], name='input')
network = conv_2d(network, 32, 3, activation = 'relu')
network = conv_2d(network, 16, 3, activation = 'relu')
network = conv_2d(network, 16, 3, activation = 'relu')
network = conv_2d(network, 32, 3, activation = 'relu')
network = conv_2d(network, 3, 3, activation = 'relu')
network = regression(
	network,
	optimizer = 'momentum',
	loss = 'categorical_crossentropy',
    name = 'target'
)
model = tflearn.DNN(
	network,
	checkpoint_path = 'saved_compression',
	max_checkpoints = 1,
	tensorboard_verbose = 2
)
if isfile('compression_video.tflearn'):
	model.load('compression_video.tflearn')
	print('[+] Model loaded')
else:
	print('[+] Training')
	model.fit(
		data, data,
		validation_set = ({'input': data}, {'target': data}),
		n_epoch = 50,
		batch_size = 1,
		shuffle = True,
		show_metric = True,
		snapshot_step = 200,
		snapshot_epoch = True
	)
	model.save('compression_video.tflearn')
	print('[+] Model Saved')
print('[+] Compressing')
compress = model.predict(data)
print(compress.shape)
compress = compress * 255. 
compressed = compress.astype('uint8')
print(compressed.dtype)
np.save('video-decoded.npy', compressed)
