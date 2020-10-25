import os
import sys
from time import time
import numpy as np
import tensorflow as tf
import pickle

from tensorflow import data
from tensorflow.keras.callbacks import LearningRateScheduler, CSVLogger
from functools import partial
from tensorflow.keras.preprocessing.image import DirectoryIterator, ImageDataGenerator

from tensorflow.keras.applications import MobileNetV2


np.random.seed(1000)

jobid = sys.argv[1]
# Directory of the images, will need to replace this as necessary
# trainDirectory = '/tmp/' + jobid + '/ramdisk/imagenet12/images/train/'
# valDirectory = '/tmp/' + jobid + '/ramdisk/imagenet12/images/sortedVal/'

trainDirectory = 'Images/train/'
valDirectory = 'Images/val/'

# Get a list of directories where their indices act as labels
cats = tf.convert_to_tensor(os.listdir(trainDirectory))

# Mean of Imagenet channels
mean = tf.stack([tf.zeros([224, 224])+103.939,
                 tf.zeros([224, 224])+116.779,
                 tf.zeros([224, 224])+123.68], axis=2)


# Function to take each file and get the image and their label
def process_train_image(path):
    # Label is the directory
    label = tf.where(cats == tf.strings.split(path, '/')[-2])[0]
    label = tf.one_hot(label[0], cats.shape[0])

    # Load image
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)

    # Scale randomly and crop image as needed
    shape = tf.shape(img)
    dim1 = tf.cast(shape[0], dtype='float32')
    dim2 = tf.cast(shape[1], dtype='float32')
    scale = tf.random.uniform([], minval=256, maxval=512, dtype='float32')

    if dim1 < scale or dim2 < scale:
        if dim1 < dim2:
            newDim1 = dim2/dim1*scale
            newDim1 = tf.math.round(newDim1)
            img = tf.image.resize(img, [newDim1, scale])
        else:
            newDim2 = dim1/dim2*scale
            newDim2 = tf.math.round(newDim2)
            img = tf.image.resize(img, [scale, newDim2])
    else:
        img = tf.image.convert_image_dtype(img, tf.float32)

    # Random horizontal flip
    img = tf.image.random_flip_left_right(img)

    # Crop to centre (note that this never pads)
    img = tf.image.resize_with_crop_or_pad(img, 224, 224)

    # Roughly approximate RGB shift (need to actually do PCA on the entire dataset)
    img = tf.image.random_hue(img, 0.1)

    # Zero mean
    img -= mean

    return img, label


# Processor for validation images
def process_val_image(path):
    # Label is the directory
    label = tf.where(cats == tf.strings.split(path, '/')[-2])[0]
    label = tf.one_hot(label[0], cats.shape[0])

    # Load image
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)

    # Scale randomly and crop image as needed
    shape = tf.shape(img)
    dim1 = tf.cast(shape[0], dtype='float32')
    dim2 = tf.cast(shape[1], dtype='float32')
    scale = 224

    if dim1 < scale or dim2 < scale:
        if dim1 < dim2:
            newDim1 = dim2/dim1*scale
            newDim1 = tf.math.round(newDim1)
            img = tf.image.resize(img, [newDim1, scale])
        else:
            newDim2 = dim1/dim2*scale
            newDim2 = tf.math.round(newDim2)
            img = tf.image.resize(img, [scale, newDim2])
    else:
        img = tf.image.convert_image_dtype(img, tf.float32)

    # Crop to centre (note that this never pads)
    img = tf.image.resize_with_crop_or_pad(img, 224, 224)

    # Zero mean
    img -= mean

    return img, label

def datasetFromDirectory(directory, batch_size, size):
    """Return a dataset flowed from a directory with basic preprocessing."""
    # Create lists of labels and image paths
    indexLabel = -1
    images = []
    labels = []
    for root, _, files in os.walk(directory):
        # Skip empty directories
        if len(files) == 0:
            continue

        indexLabel += 1
        labels += [indexLabel]*len(files)
        filePaths = [os.path.join(root, fname) for fname in files]
        images += filePaths

    # Convert image paths to tensor
    images = tf.convert_to_tensor(images)
    # Convert labels to onehot
    labels = tf.one_hot(labels, indexLabel+1)

    # Create dataset
    ds = tf.data.Dataset.from_tensor_slices((images, labels))

    # Shuffle entire dataset and cache it all
    ds = ds.shuffle(len(images)).cache()

    # Batch dataset
    ds = ds.batch(batch_size)

    # Image preprocess
    def preprocess(imgPaths, labels):
        # Load images
        def _loadRead(imgPath):
            imageFile = tf.io.read_file(imgPath)
            image = tf.image.decode_image(imageFile, channels=3)
            # TODO: Use tfrecords to preprocess with smart_resize.
            image = tf.image.resize_with_crop_or_pad(image,
                                                     target_height=size[0],
                                                     target_width=size[0])

            return image

        # Note that parallel iterations can be increased to load more images
        images = tf.map_fn(_loadRead, imgPaths, parallel_iterations=64,
                           dtype=tf.uint8)

        # Preprocess image
        images = tf.cast(images, tf.float32)
        images = tf.keras.applications.mobilenet_v2.preprocess_input(images)

        return images, labels

    ds = ds.map(preprocess, num_parallel_calls=4)

    # Add prefetching
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds

# Create datasets
# trainData = data.Dataset.list_files(trainDirectory + '*/*.JPEG')\
#     .prefetch(tf.data.experimental.AUTOTUNE)\
#     .map(process_train_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
#     .batch(128)

# valData = data.Dataset.list_files(valDirectory + '*/*.JPEG')\
#     .prefetch(tf.data.experimental.AUTOTUNE)\
#     .map(process_val_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
#     .batch(256)

#dataGen = ImageDataGenerator(preprocessing_function=preprocess_input)
#trainData = DirectoryIterator(trainDirectory,dataGen,target_size=(224, 224))
#valData = DirectoryIterator(valDirectory, dataGen, target_size=(224, 224))

trainData = datasetFromDirectory(trainDirectory, batch_size=128, size=(224, 224))
valData = datasetFromDirectory(valDirectory, batch_size=128, size=(224, 224))

# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()

# Callback to save model at beginning of every epoch in case of failure
model_path = "model_data" + "/" + jobid + "/"

if not os.path.exists(model_path):
    os.makedirs(model_path)

class SaveModelStateCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []
        self.epoch_time_end = time()

    def on_epoch_begin(self, epoch, logs=None):
        self.model.save(model_path + "checkpoint{}.h5".format(epoch))
        self.epoch_time_start = time()
        self.times.append(time() - self.epoch_time_end)

    def on_epoch_end(self, batch, logs={}):
        self.epoch_time_end = time()
        self.times.append(time() - self.epoch_time_start)

# Learning rate scheduler
def scheduler(epoch, lr):
    decay_rate = 0.7
    decay_step = 6
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr

lrcallback = LearningRateScheduler(scheduler)

# Saving history in case of failure
csv_logger = CSVLogger(model_path + jobid + "_model_history_log.csv", append=True)

model_state = SaveModelStateCallback()

# Callbacks
callbacks_list = [lrcallback, csv_logger, model_state]



# SGD optimizer
# sgd = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=False)

# Adam optimizer
adam = tf.keras.optimizers.Adam(lr = 0.001)

# Get top 5 accuracy data instead of just accuracy
top5_acc = partial(tf.keras.metrics.top_k_categorical_accuracy, k=5)
top5_acc.__name__ = 'top5_acc'


with strategy.scope():
    model = MobileNetV2(input_shape=(224,224,3), weights=None)
    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=adam, metrics=[top5_acc])

start = time()

history = model.fit(trainData, epochs=100, batch_size=128, verbose=2,
                    validation_data=valData, callbacks=callbacks_list)
end = time()

epoch_times = model_state.times

np.savetxt(model_path + jobid + 'epoch_training_times.csv', epoch_times, delimiter=',')

with open(model_path + jobid + 'trainHistoryDict', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

print("Training time is: " + str(end - start))
score = model.evaluate(valData, batch_size=128)