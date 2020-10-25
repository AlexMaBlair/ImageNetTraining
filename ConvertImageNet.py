import os
import sys
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

jobid = sys.argv[1]
# Subdirectory names of the images, easier to manipulate fo
trainDirectoryName = 'train/'
valDirectoryName = 'sortedVal/'

def preprocessDir(dirName):
    dir = '/tmp/' + jobid + '/ramdisk/imagenet12/images/' + dirName
    #dir = 'Images/' + dirName

    if os.path.exists('processed.csv'):
        with open('processed.csv') as f:
            reader = csv.reader(f)
            processed = list(reader)[0]

    else:
        processed = []

    for root, subdir, files in os.walk(dir):
        # Skip empty directories
        if len(files) == 0:
            continue

        if root in processed:
            continue

        newRoot = '/scratch/maa2/processed/ramdisk/imagenet12/images/' + dirName + os.path.basename(os.path.normpath(root))
        #newRoot = 'Preprocessed/' + root
        if not os.path.exists(newRoot):
            os.makedirs(newRoot)

        print(newRoot)

        for fname in files:
            imagePath = os.path.join(root, fname)
            imageFile = tf.io.read_file(imagePath)
            image = tf.image.decode_image(imageFile, channels=3)

            image = tf.cast(image, tf.float32)
            image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

            height, width, channels = tf.shape(image)

            if (height > width):
                image = tf.image.resize(image, (height/width * 224,224), method=tf.image.ResizeMethod.BICUBIC)
                image = tf.image.crop_to_bounding_box(image, int((height/width * 224 - 224) / 2), 0, 224, 224)
            elif (height < width):
                image = tf.image.resize(image, (224,width/height * 224), method=tf.image.ResizeMethod.BICUBIC)
                image = tf.image.crop_to_bounding_box(image, 0, int((width/height * 224 - 224) / 2), 224, 224)


            #image = tf.keras.preprocessing.image.smart_resize(image, (224,224), interpolation='bicubic')

            imgArr = image.numpy()

            # Save in new path
            newPath = os.path.join(newRoot, os.path.splitext(fname)[0])
            np.save(newPath,imgArr)

        processed.append(root)

        if os.path.exists('processed.csv'):
            os.remove('processed.csv')

        with open('processed.csv', 'w', newline='') as fp:
            csvwriter = csv.writer(fp)
            csvwriter.writerow(processed)

preprocessDir(trainDirectoryName)