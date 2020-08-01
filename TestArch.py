from SqueezeNet import SqueezeNet

import os
import sys
from time import time
import numpy as np
import tensorflow as tf
import pickle

from tensorflow import data
from tensorflow.keras.callbacks import LearningRateScheduler, CSVLogger
from functools import partial

np.random.seed(1000)

jobid = sys.argv[1]
# Directory of the images, will need to replace this as necessary
trainDirectory = '/tmp/' + jobid + '/ramdisk/imagenet12/images/train/'
valDirectory = '/tmp/' + jobid + '/ramdisk/imagenet12/images/sortedVal/'

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


# Create datasets
trainData = data.Dataset.list_files(trainDirectory + '*/*.JPEG')\
    .prefetch(tf.data.experimental.AUTOTUNE)\
    .map(process_train_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
    .batch(128)

valData = data.Dataset.list_files(valDirectory + '*/*.JPEG')\
    .prefetch(tf.data.experimental.AUTOTUNE)\
    .map(process_val_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
    .batch(256)

# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()


# Learning rate scheduler
#def scheduler(epoch):
#    if epoch < 5:
#        return 0.001
#    else:
#        return 0.00001


# lrcallback = LearningRateScheduler(scheduler)

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

# Saving history in case of failure
csv_logger = CSVLogger(model_path + jobid + "_model_history_log.csv", append=True)

model_state = SaveModelStateCallback()

# Start from checkpoint weights, if it exists
# old_jobid = 12123214
# old_chkpt_num = 4
# model = models.load_model("model_data" + "/" + old_jobid + "/checkpoint{}.h5".format(old_chkpt_num))

# SGD optimizer
sgd = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=False)

# Get top 5 accuracy data instead of just accuracy
top5_acc = partial(tf.keras.metrics.top_k_categorical_accuracy, k=5)
top5_acc.__name__ = 'top5_acc'


with strategy.scope():
    model = SqueezeNet(1000)
    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=sgd, metrics=[top5_acc])

start = time()

history = model.fit(trainData, epochs=10, batch_size=64, verbose=2,
                    validation_data=valData, callbacks=[model_state,csv_logger])
end = time()

epoch_times = model_state.times

np.savetxt(model_path + jobid + 'epoch_training_times.csv', epoch_times, delimiter=',')

with open(model_path + jobid + 'trainHistoryDict', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

print("Training time is: " + str(end - start))
score = model.evaluate(valData, batch_size=128)