from skimage import io 
import os
import numpy
from scipy import misc

def extract_images(dir,N):
    training_inputs = numpy.asarray([misc.imresize((255.0 - io.imread(dir+str(i)+'.png'))/255.0,(28,28)) for i in range(N)])
    (x,y,z) = training_inputs.shape
    training_inputs = training_inputs.reshape(x, y, z, 1)
    return training_inputs

def dense_to_one_hot(labels_dense, num_classes=104):
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(dir):
    labels = []
    with open(dir+'labels.txt','rb') as f:
        for line in f:
            labels.append(int(line.split()[0]))
    labels = numpy.asarray(labels,dtype=numpy.uint8)
    return dense_to_one_hot(labels)

def image():
    tr_dir = "dataset/train/"
    va_dir = "dataset/test/"

    train_labels = extract_labels(tr_dir)
    N = train_labels.shape[0]
    train_images = extract_images(tr_dir,N)
    
    test_labels = extract_labels(va_dir)
    N = test_labels.shape[0]
    test_images = extract_images(va_dir,N)

    print "Data reading complete"

    #train_images = train_images.reshape(17205, 784)
    #test_images = test_images.reshape(1829,784)
    return train_images,train_labels,test_images,test_labels

