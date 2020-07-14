from tensorflow import data
import tensorflow as tf
import numpy as np
import os
from PIL import Image

class IMSequence():

  def __init__(self, directory, batch_size = 256, rgbShift = 50,
  scale = (256, 512), validation = False, seed = 1234):

    self.batch_size = batch_size
    self.rgbShift = rgbShift
    self.scale = scale
    self.validation = validation
    np.random.seed(seed)

    # Get a list of class directories
    classes = os.listdir(directory)
    classes.sort()

    # Find all files in root dir
    list_ds = data.Dataset.list_files(directory + '*/*')

    # Map files to classes via file path
    def process_path(file_path):
      label = tf.strings.split(file_path, os.sep)[-2]
      return file_path, label

    # map each class to an integer
    mapping = {}
    for x in range(len(classes)):
      mapping[classes[x]] = x

    labeled_ds = list_ds.map(process_path)

    # convert tf Dataset to list of tuples
    it = labeled_ds.as_numpy_iterator()
    tmp_list = list(it)

    # convert list of tuples to np array
    string_mappings = np.array(tmp_list,dtype=str)

    # remove directories from mappings
    basedir = os.path.basename(directory)
    string_mappings = string_mappings[~(string_mappings[:, 1] == basedir)]

    # represents class labels for images
    labels = string_mappings[:,1]

    # integer representation
    for x in range(len(labels)):
      labels[x] = mapping[labels[x]]

    # x is filenames, y is onehot categories
    self.x = string_mappings[:,0]
    self.y = np.eye(len(classes))[np.array(labels,dtype=np.int8).reshape(-1)]


    # Shuffle
    rand = np.random.permutation(range(len(self.x)))
    self.x = np.array(self.x)[rand]
    self.y = self.y[rand]

    # Print information
    print('Images: ' + str(len(self.x)))
    print('Classes: ' + str(len(classes)))


  def on_epoch_end(self):
    # Shuffle
    rand = np.random.permutation(range(len(self.x)))
    self.x = np.array(self.x)[rand]
    self.y = self.y[rand, :]


  def __len__(self):
    return int(np.ceil(len(self.x) / float(self.batch_size)))


  def __getitem__(self, idx):
    images = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
    yBatch = self.y[idx * self.batch_size:(idx + 1) * self.batch_size, :]

    # Check if batch is small
    if yBatch.shape[0] < self.batch_size:
      images = images[0:yBatch.shape[0]]


    # VGG Pre-processing (taken from tf source)
    def preprocess_input(x):
      # 'RGB'->'BGR'
      x = x[..., ::-1]
      mean = [103.939, 116.779, 123.68]

      # Zero-center by mean pixel
      x[..., 0] -= mean[0]
      x[..., 1] -= mean[1]
      x[..., 2] -= mean[2]
      return x


    if self.validation:
      # Loop through images
      xBatch = np.empty((len(images), 224, 224, 3), dtype = 'int16')
      for i in range(len(images)):
        # Load image
        img = Image.open(images[i])

        # Scale and crop to centre
        if img.width > img.height: # Height is the smaller edge
          newWidth = int(np.round((img.width / img.height) * 224))
          img = img.resize((newWidth, 224), resample = Image.BICUBIC)

          # Convert to RGB as necessary
          if img.mode != 'RGB':
            img = img.convert(mode = 'RGB')

          # Convert to numpy arrays
          img = np.array(img, dtype=float)

          # Crop to centre
          start = int(np.ceil((img.shape[0] - 224) / 2))
          img = img[:, start:start + 224, :]
        elif img.width == img.height:
          img = img.resize((224, 224), resample = Image.BICUBIC)

          # Convert to RGB as necessary
          if img.mode != 'RGB':
            img = img.convert(mode = 'RGB')

          # Convert to numpy arrays
          img = np.array(img, dtype=float)
        else: # Width is the smaller edge
          newHeight = int(np.round((img.height / img.width) * 224))
          img = img.resize((224, newHeight), resample = Image.BICUBIC)

          # Convert to RGB as necessary
          if img.mode != 'RGB':
            img = img.convert(mode = 'RGB')

          # Convert to numpy arrays
          img = np.array(img, dtype=float)

          # Crop to centre
          start = int(np.ceil((img.shape[1] - 224) / 2))
          img = img[start:start + 224, :, :]

        # Save image to batch
        xBatch[i, :, :, :] = preprocess_input(img)
    else:
      # Generate random numbers for scale, flip, rgb
      randScale = self.scale[1] - np.random.randint(0, self.scale[0] + 1,
                                                    size = self.batch_size)
      randFlip = np.random.randint(0, 2, size = self.batch_size)
      randRGB = np.random.randint(-self.rgbShift, self.rgbShift + 1, size = self.batch_size)

      # Loop through images
      xBatch = np.empty((len(images), 224, 224, 3), dtype = 'int16')
      for i in range(len(images)):
        # Load the image
        img = Image.open(images[i])

        # Random scale
        targetSize = randScale[i]

        if img.width > img.height: # Height is the smaller edge
          newWidth = int(np.round((img.width / img.height) * targetSize))

          img = img.resize((newWidth, targetSize), resample = Image.BICUBIC)
        elif img.width == img.height:
          img = img.resize((targetSize, targetSize), resample = Image.BICUBIC)

        else: # Width is the smaller edge
          newHeight = int(np.round((img.height / img.width) * targetSize))

          img = img.resize((targetSize, newHeight), resample = Image.BICUBIC)

        # Convert to RGB as necessary
        if img.mode != 'RGB':
          img = img.convert(mode = 'RGB')

        # Convert to numpy arrays
        img = np.array(img, dtype=float)

        # Shift channel
        img += randRGB[i]

        # vgg preprocess
        img = preprocess_input(img)

        # Perform flip if necessary
        if randFlip[i]:
          img = np.flip(img, axis = 1)

        # Calculate random crop locations
        xStart = np.random.randint(1, img.shape[0] - 224) if img.shape[0] != 224 else 1
        yStart = np.random.randint(1, img.shape[1] - 224) if img.shape[1] != 224 else 1

        # Crop and add image to list
        xBatch[i, :, :, :] = img[xStart:xStart + 224, yStart:yStart + 224, :]

    return xBatch, yBatch


  def __iter__(self):
    for i in range(len(self.x)):
      yield [self.x[i],self.y[i]]




a = IMSequence('D:\\Current Work\\Junior College\\2Spring Semester (2020)\\Research\Python Stuff\\IMexamples', validation=True)

a.__getitem__(1)

