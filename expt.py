#!/usr/bin/python
'''
This code is used to predict the spatrial temporal data by deep learning.
Key features:
1. Residual network for temporal prediction
2. Convolutional network for spatial prediction
3. Daily grid events integration to improve temporal regularity
4. Cubic spline interpolation to improve spatial regularity
Usage: THEANO_FLAGS=mode=FAST_RUN, device=gpu, floatX=float32 python expt.py
'''
from __future__ import print_function
import cPickle as pickle
import numpy as np
import os
import time
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import metrics as metrics
from STResNet import stresnet
import FeatureGeneration
import ImgTransform

# Parameters
# Number of epoch at training stage
nb_epoch = 200
# Number of epcoj at training (cont) stage
nb_epoch_cont = 50
# Batch size
batch_size = 48
# Time space and number of time intervals each day
timeEachDay = 24
# Learning rate
lr = 0.0005
# Length of closeness dependent sequence
len_closeness = 3
# Length of periodic dependent sequence
len_period = 3
# length of trend dependent sequence
len_trend = 3
# Trend and period length
TrendInterval = 7
PeriodInterval = 1
# Number of layers of residual network
nb_residual_unit = 6
# There are multi-channel imagers: number of crime (1st channel); topic (remaining channels)
nb_channel = 1
# Test days
# days_test=14	#Temporal superresolution, use 28
days_test = 28
len_test = timeEachDay * days_test
# Partition LA into ncell intervals
ncell = 30
# Image resolutions
image_height = 32
image_width = 32
# File that stores crime data
filename = 'Crime2014_2015.csv'
# Total days data used
# ndays=200 #Train days: 3960/24=165; Test days: 336/24=14
ndays = 400  # Temporal superresolution
# Image interpolation ratio
scale = 2.0

# Build the path to stire the result and model
path_result = 'RET'
path_model = 'MODEL'
if os.path.isdir(path_result) is False:
    os.mkdir(path_result)
if os.path.isdir(path_model) is False:
    os.mkdir(path_model)


# Build the STResnet model
def build_model(external_dim):
    c_conf = (len_closeness, nb_channel, image_height, image_width) if len_closeness > 0 else None
    p_conf = (len_period, nb_channel, image_height, image_width) if len_period > 0 else None
    t_conf = (len_trend, nb_channel, image_height, image_width) if len_trend > 0 else None
    model = stresnet(c_conf=c_conf, p_conf=p_conf, t_conf=t_conf, external_dim=external_dim,
                     nb_residual_unit=nb_residual_unit)
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse])
    model.summary()
    ##Plot the network model
    # from keras.utils.visualize_util import plot
    # plot(model, to_file='ST-ResNet.png', show_shapes=True)
    return model


# The main function
if __name__ == '__main__':
    # Load data
    X_train, Y_train, X_test, Y_test, mmn, external_dim, idx_train, idx_test = FeatureGeneration.FeatureGeneration(
        len_closeness=len_closeness, len_period=len_period, len_trend=len_trend, len_test=len_test, meteorol_data=True,
        holiday_data=True, meta_data=True, ncell=ncell, ndays=ndays, timeEachDay=timeEachDay, filename=filename,
        scale=scale)

    # Build model
    print('Compiling modeling...')
    model = build_model(external_dim)
    hyperparams_name = 'c{}.p{}.t{}.resunit{}.lr{}'.format(len_closeness, len_period, len_trend, nb_residual_unit, lr)
    fname_param = os.path.join('MODEL', '{}.best.h5'.format(hyperparams_name))
    #    early_stopping=EarlyStopping(monitor='val_rmse', patience=2, mode='min')
    model_checkpoint = ModelCheckpoint(fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')

    # Train model
    print('Training model...')
    ts = time.time()
    #    history=model.fit(X_train, Y_train, nb_epoch=nb_epoch, batch_size=batch_size, validation_split=0.1, callbacks=[early_stopping, model_checkpoint], verbose=1)
    history = model.fit(X_train, Y_train, nb_epoch=nb_epoch, batch_size=batch_size, validation_split=0.1,
                        callbacks=[model_checkpoint], verbose=1)
    model.save_weights(os.path.join('MODEL', '{}.h5'.format(hyperparams_name)), overwrite=True)
    pickle.dump((history.history), open(os.path.join(path_result, '{}.history.pkl'.format(hyperparams_name)), 'wb'))
    print("\nElapsed time (training): %.3f seconds\n" % (time.time() - ts))

    # Test model
    print('Evaluating using the model that has the best loss on the validation set.')
    ts = time.time()
    model.load_weights(fname_param)
    score = model.evaluate(X_train, Y_train, batch_size=Y_train.shape[0] // timeEachDay, verbose=0)
    print('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f' % (
    score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
    score = model.evaluate(X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
    print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' % (
    score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
    print("\nElapsed time (eval): %.3f seconds\n" % (time.time() - ts))

    # Train continued
    print("Training model (cont)...")
    ts = time.time()
    fname_param = os.path.join('MODEL', '{}.cont.best.h5'.format(hyperparams_name))
    model_checkpoint = ModelCheckpoint(fname_param, monitor='rmse', verbose=0, save_best_only=True, mode='min')
    history = model.fit(X_train, Y_train, nb_epoch=nb_epoch_cont, verbose=1, batch_size=batch_size,
                        callbacks=[model_checkpoint])
    pickle.dump((history.history),
                open(os.path.join(path_result, '{}.cont.history.pkl'.format(hyperparams_name)), 'wb'))
    model.save_weights(os.path.join('MODEL', '{}_cont.h5'.format(hyperparams_name)), overwrite=True)
    print("\nElapsed time (training cont): %.3f seconds\n" % (time.time() - ts))

    # Test model
    print('Evaluating using the final model')
    score = model.evaluate(X_train, Y_train, batch_size=Y_train.shape[0] // timeEachDay, verbose=0)
    print('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f' % (
    score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
    ts = time.time()
    score = model.evaluate(X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
    print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' % (
    score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
    print("\nElapsed time (eval cont): %.3f seconds\n" % (time.time() - ts))

    # Predict on test set
    testPredict = model.predict(X_test)
    testPredict = mmn.inverse_transform(testPredict)
    testPredict2 = np.zeros((testPredict.shape[0], 1, 16, 16))
    for i in range(testPredict.shape[0]):
        testPredict2[i, 0, :, :] = ImgTransform.img_enlarge(testPredict[i, 0, :, :], 0.5, 2)

    # Test prediction
    print('testPredict2 shape: ', testPredict2.shape)
    out1 = open('TestPredict.csv', 'w')
    for i in range(testPredict2.shape[0]):
        for j in range(testPredict2.shape[2]):
            for k in range(testPredict2.shape[3]):
                out1.write('%d, %d, %d, %d' % (i + 1, j + 1, k + 1, testPredict2[i, :, j, k]))
                out1.write("\n")
    out1.close()

    # Test exact
    testExact = mmn.inverse_transform(Y_test)
    testExact2 = np.zeros((testExact.shape[0], 1, 16, 16))
    print('testExact2 shape: ', testExact2.shape)
    for i in range(testExact.shape[0]):
        testExact2[i, 0, :, :] = ImgTransform.img_enlarge(testExact[i, 0, :, :], 0.5, 2)
    out2 = open('TestExact.csv', 'w')
    for i in range(testExact2.shape[0]):
        for j in range(testExact2.shape[2]):
            for k in range(testExact2.shape[3]):
                out2.write('%d, %d, %d, %d' % (i + 1, j + 1, k + 1, testExact2[i, :, j, k]))
                out2.write("\n")
    out2.close()

    # Training prediction
    trainPredict = model.predict(X_train)
    trainPredict = mmn.inverse_transform(trainPredict)
    trainPredict2 = np.zeros((trainPredict.shape[0], 1, 16, 16))
    for i in range(trainPredict.shape[0]):
        trainPredict2[i, 0, :, :] = ImgTransform.img_enlarge(trainPredict[i, 0, :, :], 0.5, 2)
    print('trainPredict2 shape: ', trainPredict2.shape)
    out3 = open('TrainPredict.csv', 'w')
    for i in range(trainPredict2.shape[0]):
        for j in range(trainPredict2.shape[2]):
            for k in range(trainPredict2.shape[3]):
                out3.write('%d, %d, %d, %d' % (i + 1, j + 1, k + 1, trainPredict2[i, :, j, k]))
                out3.write("\n")
    out3.close()

    # Training exact
    trainExact = mmn.inverse_transform(Y_train)
    trainExact2 = np.zeros((trainExact.shape[0], 1, 16, 16))
    print('trainExact2 shape: ', trainExact2.shape)
    for i in range(trainExact.shape[0]):
        trainExact2[i, 0, :, :] = ImgTransform.img_enlarge(trainExact[i, 0, :, :], 0.5, 2)
    out4 = open('TrainExact.csv', 'w')
    for i in range(trainExact2.shape[0]):
        for j in range(trainExact2.shape[2]):
            for k in range(trainExact2.shape[3]):
                out4.write('%d, %d, %d, %d' % (i + 1, j + 1, k + 1, trainExact2[i, :, j, k]))
                out4.write("\n")
    out4.close()

    # Plot figures
    for i in range(testPredict2.shape[0]):
        imgplot = plt.imshow(testPredict2[i, 0, :, :])
        plt.pause(0.01)
    plt.colorbar()
    plt.show()

    # Save predicted image to file, this is done through reading the prediction and exact from file

#    plt.figure(1)
#    imgplot=plt.imshow(testPredict[-2, 0, :, :])
#    plt.colorbar()
#    testExact=mmn.inverse_transform(Y_test)
#    print('Test Predict shape: ', testPredict.shape)
#    print('Test Exact shape: ', testExact.shape)
#    plt.figure(2)
#    imgplot=plt.imshow(testExact[-2, 0, :, :])
#    plt.colorbar()
#    plt.show()
