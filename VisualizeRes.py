#!/usr/bin/python
'''
This code used to visualize the results, and transform cumulative data to hourly data. Visualize again
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Bucket data
def BucketData(filename='TestPredict.csv', ncell=16, ndays=14, timeEachDay=24):
    df = pd.read_csv(filename, header=None)
    df.columns = ['Time', 'X', 'Y', 'NumEvents']
    Time = df['Time']
    X = df['X']
    Y = df['Y']
    NumEvents = df['NumEvents']

    Events = np.zeros((Time.max(), 1, X.max(), Y.max()))
    for i in range(len(Time) / (ncell * ncell)):
        for j in range(ncell):
            Events[i, 0, j, :] = NumEvents[i * ncell * ncell + j * ncell:i * ncell * ncell + j * ncell + ncell]
    Events[Events < 0] = 0
    print('Events shape: ', Events.shape)
    return Events


# This function is used to transform the cumulative data to hourly data
def DataTransform(Events, TimeEachDay=24):
    Events2 = np.zeros(Events.shape)
    ndays = Events2.shape[0] / TimeEachDay
    for i in range(ndays):
        for j in range(TimeEachDay):
            if j == 0:
                Events2[i * TimeEachDay + j, :, :, :] = Events[i * TimeEachDay + j, :, :, :]
                Events2[Events2 < 0] = 0
            else:
                Events2[i * TimeEachDay + j, :, :, :] = Events[i * TimeEachDay + j, :, :, :] - Events2[
                                                                                               i * TimeEachDay + j - 1,
                                                                                               :, :, :]
                Events2[Events2 < 0] = 0
    return Events2


# The main function
if __name__ == '__main__':
    # Events=BucketData(filename='TestPredict.csv', ncell=16, ndays=14, timeEachDay=24)
    Events = BucketData(filename='TestExact.csv', ncell=16, ndays=14, timeEachDay=24)
    # Events=BucketData(filename='TrainPredict.csv', ncell=16, ndays=165, timeEachDay=24)
    # Events=BucketData(filename='TrainExact.csv', ncell=16, ndays=165, timeEachDay=24)

    # Transform data from cumulative to hourly intensity
    Events2 = DataTransform(Events, TimeEachDay=24)

    # Visualize data
    for i in range(Events2.shape[0]):
        imgplot = plt.imshow(Events2[i, 0, :, :])
        plt.pause(0.01)
    # plt.colorbar()
    plt.show()

    # TODO: save images to gif moive
