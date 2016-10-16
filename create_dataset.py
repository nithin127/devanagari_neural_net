import glob
import numpy
import collections
from scipy import misc

from tensorflow.python.framework import dtypes

NUM_LABELS = 104

def dense_to_one_hot(labels_dense, num_classes = NUM_LABELS):
  """Convert class labels from scalars to one-hot vectors."""
  # This is taken from tensorflow.contrib.learn.python.learn.datasets.mnist
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def importDataset(data_type, dataset_directory, one_hot = True):
  images = []
  labels = []
  # Creating image matrix
  filenames = []
  for item in glob.glob(dataset_directory + data_type + '/*.png'):
    filenames.append(item)
  filenames = sorted(filenames, key = len) # So that '9.png' comes before '11.png'
  for filename in filenames:
  	im = misc.imresize(misc.imread(filename),(28,28))
  	images.append(im)
  images = (255.0 - numpy.array(images))/255.0 # Highly important, apparently
  images = images.reshape(-1,images.shape[1],images.shape[2],1)
  # Creating label list
  for filename in glob.glob(dataset_directory + data_type + '/labels.txt'):
  	label_file = open(filename)
  labels = label_file.readlines()
  labels = [int(x[:-1]) for x in labels]
  labels = numpy.array(labels).astype('uint8')
  # Convert labels to one_hot encoding if necessary
  if one_hot:
    return images, dense_to_one_hot(labels,NUM_LABELS)
  else:
    return images, labels

def getDataset(dataset_directory = '/Users/nithinvasisth/Documents/advanced_ml/asgn/devnagari/dataset/',
               validation = False, one_hot = True):

  test, test_labels = importDataset('test', dataset_directory)
  train, train_labels = importDataset('train', dataset_directory)
  #train, train_labels = importDataset('sample_train', dataset_directory)
  #test, test_labels = importDataset('sample_test', dataset_directory)
  #test, test_labels = importDataset('sample', dataset_directory, one_hot)
  #train, train_labels = importDataset('sample', dataset_directory, one_hot)

  if not validation:
    return train, train_labels, test, test_labels
  else:
    # Shuffling the training dataset
    perm = numpy.arange(train.shapep[0])
    numpy.random.shuffle(perm)
    train = train[perm]
    train_labels = train_labels[perm]
    # Distributing 10% of train into validation
    valid = train[int(round(train.shape[0]*0.9)):,:]
    train2 = train[:int(round(train.shape[0]*0.9)),:]
    valid_labels = train_labels[int(round(train.shape[0]*0.9)):]
    train_labels2 = train_labels[:int(round(train.shape[0]*0.9))]

    return train2, train_labels2, valid, valid_labels, test, test_labels
