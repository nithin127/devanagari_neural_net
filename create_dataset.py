import glob
import numpy as np
from scipy import misc
import cPickle as pickle

dataset_directory = '/Users/nithinvasisth/Documents/advanced_ml/asgn/devnagari/dataset/'

def getDataset(data_type):
	images = []
	labels = []
	# Creating image matrix
	for filename in glob.glob(dataset_directory + data_type + '/*.png'):
		im = misc.imread(filename)
		im.resize([1,im.shape[0]*im.shape[1]]) #assuming a 2D input
		images.append(im[0])
	images = np.array(images)
	# Creating label list
	for filename in glob.glob(dataset_directory + data_type + '/labels.txt'):
		label_file = open(filename)
	labels = label_file.readlines()
	labels = [int(x[:-1]) for x in labels]
	# Nothing
	return images, np.array(labels)


test, test_labels = getDataset('test')
train, train_labels = getDataset('train')

with open("train.pickle", "wb") as output_file:
	pickle.dump(train, output_file)
with open("test.pickle", "wb") as output_file:
	pickle.dump(test, output_file)

with open("train_labels.pickle", "wb") as output_file:
	pickle.dump(train_labels, output_file)
with open("test_labels.pickle", "wb") as output_file:
	pickle.dump(test_labels, output_file)

# Including validation dataset

valid = train[int(round(train.shape[0]*0.9)):,:]
train = train[:int(round(train.shape[0]*0.9)),:]
valid_labels = train_labels[int(round(train.shape[0]*0.9)):]
train_labels = train_labels[:int(round(train.shape[0]*0.9))]


with open("train_with_validation.pickle", "wb") as output_file:
	pickle.dump(train, output_file)
with open("valid.pickle", "wb") as output_file:
	pickle.dump(valid, output_file)

with open("train_labels_with_validation.pickle", "wb") as output_file:
	pickle.dump(train_labels, output_file)
with open("valid_labels.pickle", "wb") as output_file:
	pickle.dump(valid_labels, output_file)



