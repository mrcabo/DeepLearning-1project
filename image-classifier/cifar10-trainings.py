from math import ceil
from keras import applications
from keras.optimizers import Adam, SGD
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras import utils
from keras.datasets import cifar10
import numpy as np
import os
import matplotlib.pyplot as plt

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
from keras.models import Sequential, Model, load_model
from keras import backend as K


def printGraph(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig('model-accuracy.png')
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig('model-loss.png')


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    # lr = 0.0001  # For VGG16, 1e-3 was too high. start from this value.
    if epoch > 90:
        lr *= 0.5e-3
    elif epoch > 80:
        lr *= 1e-3
    elif epoch > 60:
        lr *= 1e-2
    elif epoch > 40:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def build_model(input_shape, pre_trained_weights, num_classes, dropout=False, model='resnet'):
    if model == 'densenet121':
        base_model = applications.DenseNet121(weights=pre_trained_weights, include_top=False, input_shape=input_shape)
    elif model == 'vgg16':
        base_model = applications.VGG16(weights=pre_trained_weights, include_top=False, input_shape=input_shape)
    else:
        base_model = applications.ResNet50(weights=pre_trained_weights, include_top=False, input_shape=input_shape)
    if pre_trained_weights == 'imagenet':
        for layer in base_model.layers:
            layer.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    if dropout:
        x = Dropout(0.3)(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(num_classes, activation='softmax')(x)
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def run_model(x_train_data, y_train_data, x_test_data, y_test_data, model_name, input_shape, pre_trained_weights,
              num_classes, batch_size=32, epochs=100, optimizer='Adam', loss_typename='categorical_crossentropy',
              dropout=False):
    model = build_model(input_shape, pre_trained_weights, num_classes, dropout, model=model_name)
    if optimizer == 'SGD':
        opt = SGD(lr=lr_schedule(0))
    elif optimizer == 'SGD_nesterov':
        opt = SGD(lr=lr_schedule(0), momentum=0.9, decay=1e-4, nesterov=True)
    else:
        opt = Adam(lr=lr_schedule(0))

    if loss_typename == 'mse':
        loss = 'mse'
    else:
        loss = 'categorical_crossentropy'

    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
    model.summary()
    with_gpu = K.tensorflow_backend._get_available_gpus()
    print('If not empty, the code is using GPU:')
    print(with_gpu)

    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    callbacks = [lr_reducer, lr_scheduler]

    num_train_steps = ceil(len(x_train_data) / batch_size)
    num_test_steps = ceil(len(x_test_data) / batch_size)

    # Run training, with or without data augmentation.
    if not data_augmentation:
        print('Not using data augmentation.')
        history = model.fit(x_train_data, y_train_data,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(x_test_data, y_test_data),
                            shuffle=True, verbose=2,
                            callbacks=callbacks)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # epsilon for ZCA whitening
            zca_epsilon=1e-06,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # set range for random shear
            shear_range=0.,
            # set range for random zoom
            zoom_range=0.,
            # set range for random channel shifts
            channel_shift_range=0.,
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            # value used for fill_mode = "constant"
            cval=0.,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False,
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train_data)

        # Fit the model on the batches generated by datagen.flow().
        history = model.fit_generator(datagen.flow(x_train_data, y_train_data, batch_size=batch_size),
                                      validation_data=(x_test_data, y_test_data),
                                      epochs=epochs, verbose=2, workers=4,
                                      steps_per_epoch=num_train_steps, callbacks=callbacks)

    printGraph(history)

    # Score trained model.
    scores = model.evaluate(x_test_data, y_test_data, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])


def print_params(model_name, batch_size, epochs, optimizer, loss, dropout, data_augmentation, pre_trained):
    print('Training with:')
    print('Network model name: {}'.format(model_name))
    print('Batch size: {}'.format(batch_size))
    print('Epochs: {}'.format(epochs))
    print('Optimizer: {}'.format(optimizer))
    print('Loss: {}'.format(loss))
    print('Dropout: {}'.format(dropout))
    print('Data augmentation: {}'.format(data_augmentation))
    print('Pre-trained: {}'.format(pre_trained))

# Training parameters

model_name = 'resnet'
# model_name = 'densenet121'
# model_name = 'vgg16'
batch_size = 64
epochs = 100
# optimizer = 'SGD'
# optimizer = 'SGD_nesterov'
optimizer = 'Adam'
loss = 'categorical_crossentropy'
dropout = True
data_augmentation = True # TODO: make it with data augmentation, should be noticeable for larger networks.. We will need to use fit_generator
pre_trained = False
num_classes = 10

if pre_trained:
    pre_trained_weights = 'imagenet'
else:
    pre_trained_weights = None

# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True

print_params(model_name, batch_size, epochs, optimizer, loss, dropout, data_augmentation, pre_trained)

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# Convert class vectors to binary class matrices.
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

run_model(x_train, y_train, x_test, y_test, model_name, input_shape, pre_trained_weights, num_classes, batch_size,
          epochs, optimizer, dropout=dropout)
