# ## Set training parameters

n_loops = 10
epoch_by_loop_first = 500
epoch_by_loop = 200
training_batch_size = [20]*10
training_range = [0, 80]
validation_batch_size = [50]*10
validation_range = [80, 100]
lr = 0.003
lr_decay = 0.999
drop = 0.3

# ## Extract power histogram slices


import librosa
import numpy as np
import scipy.misc
import os
from time import time

start = time()

def power_hist(sound, sr=22050, freq_samples=128, sample_freq=50):
	"""compute the power frequency histogram of a sound"""
	D = librosa.stft(sound, n_fft=int(freq_samples*2-1), hop_length=int(sr/sample_freq))
	return librosa.amplitude_to_db(D, ref=np.max)

def extract_random_slice(hist, width=128):
	"""extract a random slice from an hisrogram with desired length"""
	start = np.random.randint(np.shape(hist)[1] - width)
	return hist[:,start:start+width]

def get_file_path(num, genre):
	"""find the relative path of a given track"""
	return './db/music/'+genre+'/'+genre+'.'+'{:05d}'.format(num)+'.au'

def add2batch(batch, x):
	"""concatenate 2 batches"""
	if np.ndim(x) != 4:
		dim = [-1]+list(np.shape(x))+[1]
	else:
		dim = np.shape(x)
	return np.concatenate((batch, x.reshape(dim) ))

def load_file(num, genre):
	"""open audio file with librosa"""
	y, sr = librosa.load(get_file_path(num, genre))
	return y, sr

def extract_random_batch_in_slice(num, genre, _=0):
	"""get a random histogram from a given track"""
	batch = np.load('./db/slices/{}/{}{:04d}.npy'.format(genre, genre, num))
	return batch

def extract_random_batch_in_file(num, genre, N):
	"""get a batch of hostograms from a given track"""
	y, sr = load_file(num, genre)
	hist = power_hist(y)

	curr = extract_random_slice(hist)
	batch = np.zeros(np.shape(curr))
	batch = batch.reshape([-1, np.shape(batch)[0], np.shape(batch)[1], 1])
	batch = add2batch(batch, curr)
	batch = batch[1:, :, :, :]

	for i in range(1,N):
		curr = extract_random_slice(hist)
		batch = add2batch(batch, curr)

	return batch

def extract_random_batch(N, genre, range_sel=[0,100], n_by_track=10):
	"""extract a complete batch from randomly chosen histograms"""
	n_tracks = int(np.ceil(N/n_by_track))
	num = np.random.randint(range_sel[0], range_sel[1])
	batch = extract_random_batch_in_slice(num, genre, N)
	for _ in range(1, n_tracks):
		num = np.random.randint(range_sel[0], range_sel[1])
		batch = add2batch(batch, extract_random_batch_in_slice(num, genre, n_by_track))

	labels = generate_labels(N, genre)

	return batch[:N, :, :, :], labels

def generate_labels(size, genre):
	"""create the label vector"""
	labels = np.zeros((1,10))
	labels[0,inv_dic[genre]] = 1
	labels = np.matlib.repmat(labels, size, 1)
	return labels

def imsave_batch(batch, dir='', name='batch_im'):
	"""save the content of a batch of histogram"""
	if dir == '':
		n = np.random.randint(99999)
		os.mkdir("./save/batchsave_{:05d}".format(n))
	for i in range(np.shape(batch)[0]):
		scipy.misc.imsave("temp.jpg", batch[i, :, :, 0])
		if dir == '':
			os.rename("temp.jpg", "./save/batchsave_{:05d}/{}}{:04d}.jpg".format(n, name,i))
		else:
			os.rename("temp.jpg", "./db/slices/{}/{}{:04d}.jpg".format(dir, name, i))

def get_complete_set(n_by_genre, range_sel=[0,100], n_by_track=10):
	"""create a complete dataset of histograms"""
	genre = 0
	batch, labels = extract_random_batch(n_by_genre[genre], lab_dic[genre], range_sel, n_by_track)
	for genre in range(1, 10):
		curr, lab = extract_random_batch(n_by_genre[genre], lab_dic[genre], range_sel, n_by_track)
		batch = add2batch(batch, curr)
		labels = np.concatenate((labels, lab))

	return batch, labels


lab_dic = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
inv_dic = {lab_dic[i]: i for i in range(len(lab_dic))}
lab_dic = {i: lab_dic[i] for i in range(len(lab_dic))}

# X_val, y_val = get_complete_set(validation_batch_size, validation_range, n_by_track=2)

# print(np.shape(X_val))
# print(np.shape(y_val))

# ## Build database

# for genre in inv_dic:
# 	print(genre)
# 	for num in range(100):
# 		print(num)
# 		batch = extract_random_batch_in_file(num, genre, 10)
# 		np.save('temp.npy', batch)
# 		os.rename('temp.npy', './db/slices/{}/{}{:04d}.npy'.format(genre, genre, num))

# ## Build the CNN


import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.optimizers import RMSProp
from sklearn.metrics import confusion_matrix

# Building graph

# convolutional layers
network = input_data(shape=[None, 128, 128, 1], name='input')
network = conv_2d(network, 64, [2, 2], 2, activation='elu', regularizer="L2", weights_init="Xavier", bias_init="Xavier")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 128, [2, 2], 2, activation='elu', regularizer="L2", weights_init="Xavier", bias_init="Xavier")
network = max_pool_2d(network, 2)
network = conv_2d(network, 256, [2, 2], 2, activation='elu', regularizer="L2", weights_init="Xavier", bias_init="Xavier")
network = max_pool_2d(network, 2)
network = conv_2d(network, 512, [2, 2], 2, activation='elu', regularizer="L2", weights_init="Xavier", bias_init="Xavier")
network = max_pool_2d(network, 2)

# fully conneted layers
network = local_response_normalization(network)
network = fully_connected(network, 1024, activation='elu')
network = dropout(network, drop)
network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer=RMSProp(learning_rate=0.1, decay=0.999), loss='categorical_crossentropy', name='target')

# Training

model_name = 'dcnn_good_{}-{}_{}_{}-{}_{}'.format(lr, lr_decay, np.sum(training_batch_size), epoch_by_loop_first, epoch_by_loop, drop)
model = tflearn.DNN(network, tensorboard_verbose=1, tensorboard_dir='./tmp/'+model_name)


os.mkdir("models/"+model_name)


X_val, y_val = get_complete_set(validation_batch_size, validation_range)

for l in range(n_loops):
	X_batch, y_batch = get_complete_set(training_batch_size, training_range)
	if l == 0:
		epoch = epoch_by_loop_first
	else:
		epoch = epoch_by_loop

	for e in range(epoch):

		model.fit({'input': X_batch}, {'target': y_batch}, n_epoch=1,
	           validation_set=({'input': X_val}, {'target': y_val}),
				snapshot_step=1, show_metric=True, run_id='DCNN')

		if e%50 == 0:
			model.save('./models/{}/loop{}_epoch_{}.tflearn'.format(model_name, l+1, e+1))
			pred = model.predict({'input': X_val})
			pred = np.argmax(pred, axis=1)
			true_lab = np.argmax(y_val, axis=1)
			print(confusion_matrix(pred, true_lab))

print('Training finished in {:.0f}s'.format(time()-start))

Y = []
for genre in inv_dic:
	val_set = extract_random_batch_in_file(99, genre, 100)
	print('\n\n' + genre)
	pred = model.predict({'input': val_set})
	meas = np.argmax(pred, axis=1)
	print(meas)
	Y += [[m, inv_dic[genre]] for m in meas]
	res = np.mean(pred, axis=0)
	print({lab_dic[i]: res[i] for i in range(10)})
	print('most likely : {}'.format(lab_dic[np.argmax(pred)]))
print(Y)