"""
Adapted from keras example cifar10_cnn.py
Train ResNet-18 on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10.py
"""
from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from managedata import load_2class_data
import numpy as np
import resnet
import datetime
import matplotlib.pyplot as plt


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()  # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc') # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
csv_logger = CSVLogger('resnet18.csv')
nb_classes = 2

resnetnum = 18
batch_size = 20
nb_epoch = 25
data_augmentation = True
opt = 'adam'
myloss = 'categorical_crossentropy'

# input image dimensions
img_rows, img_cols = 512, 512
# The CIFAR10 images are RGB.
img_channels = 1

# The data, shuffled and split between train and test sets:
(X_train, y_train_grading, y_train_staging), (X_test, y_test_grading, y_test_staging) = load_2class_data()
# Convert class vectors to binary class matrices.
Y_train_grading = np_utils.to_categorical(y_train_grading, nb_classes)
Y_test_grading = np_utils.to_categorical(y_test_grading, nb_classes)
Y_train_staging = np_utils.to_categorical(y_train_staging, nb_classes)
Y_test_staging = np_utils.to_categorical(y_test_staging, nb_classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# subtract mean and normalize
mean_image = np.mean(X_train, axis=0)
X_train -= mean_image
X_test -= mean_image
X_train /= 128.
X_test /= 128.

Grading_model = resnet.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)
Staging_model = resnet.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)
Grading_model.compile(loss=myloss,
              optimizer=opt,
              metrics=['accuracy'])
Staging_model.compile(loss=myloss,
              optimizer=opt,
              metrics=['accuracy'])

Grading_history = LossHistory()
Staging_history = LossHistory()

if not data_augmentation:
    print('Not using data augmentation.')
    Grading_model.fit(X_train, Y_train_grading,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_split=0.15,  # (X_test, Y_test),
              shuffle=True,
              callbacks=[lr_reducer, early_stopper, csv_logger, Grading_history])
    Staging_model.fit(X_train, Y_train_staging,
                      batch_size=batch_size,
                      nb_epoch=nb_epoch,
                      validation_split=0.15,  # (X_test, Y_test),
                      shuffle=True,
                      callbacks=[lr_reducer, early_stopper, csv_logger, Staging_history])
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        # featurewise_center=True,  # set input mean to 0 over the dataset
        # samplewise_center=True,  # set each sample mean to 0
        # featurewise_std_normalization=True,  # divide inputs by std of the dataset
        # samplewise_std_normalization=True,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        # rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
        # width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        # height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(X_train)

    # Fit the model on the batches generated by datagen.flow().
    Grading_model.fit_generator(datagen.flow(X_train, Y_train_grading, batch_size=batch_size),
                        steps_per_epoch=X_train.shape[0] // batch_size,
                        # validation_data=(X_test, Y_test),
                        epochs=nb_epoch, verbose=1, max_q_size=100,
                        callbacks=[lr_reducer, early_stopper, csv_logger, Grading_history])
    Staging_model.fit_generator(datagen.flow(X_train, Y_train_staging, batch_size=batch_size),
                                steps_per_epoch=X_train.shape[0] // batch_size,
                                # validation_data=(X_test, Y_test),
                                epochs=nb_epoch, verbose=1, max_q_size=100,
                                callbacks=[lr_reducer, early_stopper, csv_logger, Staging_history])

Grading_train_result = Grading_model.evaluate(X_train, Y_train_grading)
Grading_test_result = Grading_model.evaluate(X_test, Y_test_grading)
Staging_train_result = Staging_model.evaluate(X_train, Y_train_staging)
Staging_test_result = Staging_model.evaluate(X_test, Y_test_staging)

print('Grading_Train = ', Grading_train_result)
print('Grading_Test = ', Grading_test_result)
print('Staging_Train = ', Staging_train_result)
print('Staging_Test = ', Staging_test_result)

now = datetime.datetime.now()
otherStyleTime = now.strftime("%Y-%m-%d %H:%M:%S")

Grading_model.save('Grading_model_weights/' + str(round(Grading_test_result[1], 3))
           + '-resnet' + str(resnetnum)
           + '-' + str(otherStyleTime)
           + '-' + str(batch_size)
           + '-' + str(nb_epoch)
           + '-' + str(data_augmentation)
           + '-' + str(opt)
           + '-' + str(myloss) + '.h5')
Staging_model.save('Staging_model_weights/' + str(round(Staging_test_result[1], 3))
           + '-resnet' + str(resnetnum)
           + '-' + str(otherStyleTime)
           + '-' + str(batch_size)
           + '-' + str(nb_epoch)
           + '-' + str(data_augmentation)
           + '-' + str(opt)
           + '-' + str(myloss) + '.h5')
# json_string = model.to_json()
# open('model_json/' + str(round(test_result[1], 3)) + '.json', 'w').write(json_string)
print('\nSuccessfully saved as model')
Grading_history.loss_plot('epoch')
Staging_history.loss_plot('epoch')
