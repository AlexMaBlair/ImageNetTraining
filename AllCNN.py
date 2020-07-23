import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, GlobalAveragePooling2D

def All_CNN():
    model = Sequential()

    # Define initializers
    glorotInit = tf.keras.initializers.glorot_uniform()
    truncNormInit = tf.keras.initializers.TruncatedNormal(0, 0.005)
    constantInit = tf.keras.initializers.Constant(0.1)

    # Define regularizers
    l2Reg = tf.keras.regularizers.l2(5e-4)

    # SGD optimizer
    sgd = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=False)

    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=(224, 224, 3), kernel_size=(11, 1), strides=(4, 4),
                     padding='same', name='Input', kernel_initializer=glorotInit,
                     kernel_regularizer=l2Reg, bias_initializer=constantInit))

    # .2 dropout for inputs, .5 otherwise
    model.add(Dropout(0.2))

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu',
                     kernel_initializer=glorotInit, kernel_regularizer=l2Reg, bias_initializer=constantInit))

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=96, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu',
                     kernel_initializer=glorotInit, kernel_regularizer=l2Reg, bias_initializer=constantInit))

    model.add(Dropout(0.5))

    # 4th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu',
                     kernel_initializer=glorotInit, kernel_regularizer=l2Reg, bias_initializer=constantInit))

    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu',
                     kernel_initializer=glorotInit, kernel_regularizer=l2Reg, bias_initializer=constantInit))

    # 6th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu',
                     kernel_initializer=glorotInit, kernel_regularizer=l2Reg, bias_initializer=constantInit))

    model.add(Dropout(0.5))

    # 7th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                     kernel_initializer=glorotInit, kernel_regularizer=l2Reg, bias_initializer=constantInit))

    # 8th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu',
                     kernel_initializer=glorotInit, kernel_regularizer=l2Reg, bias_initializer=constantInit))

    # 9th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu',
                     kernel_initializer=glorotInit, kernel_regularizer=l2Reg, bias_initializer=constantInit))

    model.add(Dropout(0.5))

    # 10th Convolutional Layer
    model.add(Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                     kernel_initializer=glorotInit, kernel_regularizer=l2Reg, bias_initializer=constantInit))

    # 11th Convolutional Layer
    model.add(Conv2D(filters=1024, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu',
                     kernel_initializer=glorotInit, kernel_regularizer=l2Reg, bias_initializer=constantInit))

    # 12th Convolutional Layer
    model.add(Conv2D(filters=1000, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu',
                     kernel_initializer=glorotInit, kernel_regularizer=l2Reg, bias_initializer=constantInit))

    # Global average pooling
    model.add(GlobalAveragePooling2D())

    # Output Layer
    model.add(Dense(1000, activation='softmax', name='Classification'))

    return model
