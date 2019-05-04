from keras import applications
from keras.optimizers import Adam, SGD
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

def build_resnet(input_shape, pre_trained_weights, num_classes):
    base_model = applications.ResNet50(weights=pre_trained_weights, include_top=False, input_shape=input_shape)
    if pre_trained_weights == 'imagenet':
        for layer in base_model.layers:
            layer.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # x = Dropout(0.7)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def build_densenet121(input_shape, pre_trained_weights, num_classes):
    base_model = applications.DenseNet121(weights=pre_trained_weights, include_top=False, input_shape=input_shape)
    if pre_trained_weights == 'imagenet':
        for layer in base_model.layers:
            layer.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # x = Dropout(0.7)(x)
    # let's add a fully-connected layer
    # x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(num_classes, activation='softmax')(x)
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def run_model(x_train_data, y_train_data, x_test_data, y_test_data, model_name, input_shape, pre_trained_weights,
              num_classes, batch_size=32, epochs=100, optimizer='Adam', loss_typename='categorical_crossentropy'):
    if model_name == 'densenet121':
        model = build_densenet121(input_shape, pre_trained_weights, num_classes)
    else:
        model = build_resnet(input_shape, pre_trained_weights, num_classes)

    if optimizer == 'SGD':
        opt = SGD(lr=0.001, momentum=0.9, decay=1e-3, nesterov=True)
    else:
        opt = Adam(lr=0.0001)

    if loss_typename == 'mse':
        loss = 'mse'
    else:
        loss = 'categorical_crossentropy'

    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

    with_gpu = K.tensorflow_backend._get_available_gpus()
    print('If not empty, the code is using GPU:')
    print(with_gpu)

    history = model.fit(x_train_data, y_train_data, epochs=epochs, batch_size=batch_size,
              validation_data=(x_test_data, y_test_data))
    printGraph(history)

    preds = model.evaluate(x_test_data, y_test_data)
    print("Loss = " + str(preds[0]))
    print("Test Accuracy = " + str(preds[1]))
    model.summary()


model_name = 'densenet121'
# model_name = 'resnet'

if model_name == 'densenet121':  # For densenet - https://arxiv.org/pdf/1608.06993.pdf
    batch_size = 64
    epochs = 80
    optimizer = 'SGD'
else:
    batch_size = 64  # orig paper trained all networks with batch_size=128
    epochs = 80
    optimizer = 'Adam'

# Training parameters
data_augmentation = False  # TODO: make it with data augmentation, should be noticeable for larger networks.. We will need to use fit_generator
num_classes = 10
pre_trained = False

if pre_trained:
    pre_trained_weights = 'imagenet'
else:
    pre_trained_weights = None

# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True

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

run_model(x_train, y_train, x_test, y_test, model_name, input_shape, pre_trained_weights,
          num_classes, batch_size, epochs, optimizer)
