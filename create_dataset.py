import glob
import numpy
import collections
from scipy import misc

from tensorflow.python.framework import dtypes

NUM_LABELS = 104

class DataSet(object):
  # This class is a modification of class 'Dataset' in tensorflow.contrib.learn.python.learn.datasets.mnist
  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32):

    """Construct a DataSet. one_hot arg is used only if fake_data is true.  `dtype` can be either 
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into `[0, 1]` """

    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
    
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 1024
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]

def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  # This is taken from tensorflow.contrib.learn.python.learn.datasets.mnist
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def importDataset(data_type, dataset_directory, one_hot = False):
  images = []
  labels = []
  # Creating image matrix
  filenames = []
  for item in glob.glob(dataset_directory + data_type + '/*.png'):
    filenames.append(item)
  filenames = sorted(filenames, key = len) # So that '9.png' comes before '11.png'
  for filename in filenames:
  	im = misc.imresize(misc.imread(filename),0.1)
  	images.append(im)
  images = numpy.array(images)
  images = images.reshape(-1,images.shape[1]*images.shape[2])
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
               validation = False, one_hot = False):

  test, test_labels = importDataset('test', dataset_directory)
  train, train_labels = importDataset('train', dataset_directory)
  #train, train_labels = importDataset('sample_train', dataset_directory)
  #test, test_labels = importDataset('sample_test', dataset_directory)
  #test, test_labels = importDataset('sample', dataset_directory, one_hot)
  #train, train_labels = importDataset('sample', dataset_directory, one_hot)

  if not validation:
    Datasets = collections.namedtuple('Datasets', ['train', 'test'])
    train = DataSet(train, train_labels)
    test = DataSet(test, test_labels)
    return Datasets(train=train, test=test)
  else:
    Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
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
    # Embedding the information into DataSet class
    train = DataSet(train2, train_labels2)
    valid = DataSet(valid, valid_labels)
    test = DataSet(test, test_labels)
    return Datasets(train=train, validation=valid, test=test)
