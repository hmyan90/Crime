#!/usr/bin/python
'''
This code is used to generate the features and label for training and testing set
'''
from __future__ import print_function
import numpy as np
from minmax_normalization import MinMaxNormalization
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import operator
import math
import pandas as pd
import scipy.ndimage


# Resize the image
# order=1: bilinear interpolation
# order=2: cubic interpolation
# factor: enlarged ratio
def img_enlarge(img, factor=2.0, order=1):
    return scipy.ndimage.zoom(img, factor, order=order)


# Generate the corresponding time to the number of events: periodic
# timeInterval_EachDay: the number of slots of each day
# numDays: total number of days
# ndays: the number of days in the testing set
def GenerateTimeFeatures(timeEachDay=24, numDays=730, ndays=200):
    Time = np.zeros(timeEachDay * numDays)
    count = 0
    for i in range(numDays):
        for j in range(timeEachDay):
            Time[count] = j
            count += 1
    Time = np.asarray(Time)
    # Normalization
    Time = Time / float(Time.max())
    num = ndays * timeEachDay
    return Time[-num:]


# Load holiday features, encoded to 0 or 1. Holiday:1, other 0
def load_holiday(timeEachDay=24, numDays=730, ndays=200, fname='Holiday.txt'):
    f = open(fname, 'r')
    holidays = f.readlines()
    holidays = set([h.strip() for h in holidays])
    H = np.zeros(timeEachDay * numDays)
    for i in holidays:
        ii = int(i) - 1
        for j in range(timeEachDay):
            H[ii * timeEachDay + j] = 1
    H = np.asarray(H)
    num = ndays * timeEachDay
    return H[-num:]


# Load meteoral features
def load_meteorol(timeEachDay=24, numDays=730, ndays=200, fname='WeatherFeature.csv'):
    df = pd.read_csv(fname)
    Temperature = df['Temperature']
    WindSpeed = df[' WindSpeed']
    Events = df[' Event']
    TE = np.asarray(Temperature)  # Temperature
    WS = np.asarray(WindSpeed)  # Wind speed
    EV = np.asarray(Events)  # Events, rain, etc

    num = ndays * timeEachDay
    TE1 = TE[-num:]
    WS1 = WS[-num:]
    EV1 = EV[-num:]
    merge_data = np.vstack([TE1, WS1, EV1])
    merge_data = np.transpose(merge_data)
    # print('merge data shape: ', merge_data.shape)
    return merge_data


# Bucket data, and localize to a 16 X 16 grid
# Put the number of events in each cell at each time into buckets
# ndays: the number of days' data used
# timespace: the space of each time interval
def BucketData(filename='Crime2014_2015.csv', ncell=30, ndays=200, timeEachDay=24):
    df = pd.read_csv(filename)
    OCC_DT = df['OCC_DT']
    # OCC_TO_DT = df['OCC_TO_DT']
    TIME = df['TIME']
    # TO_TIME = df['TO_TIME']
    LAT = df['LAT']
    LON = df['LON']
    # CODE = df['CODE']

    # First get the bounding box of the crime location
    LAT = np.asarray(LAT)
    LON = np.asarray(LON)
    minLAT = LAT.min()
    maxLAT = LAT.max()
    minLON = LON.min()
    maxLON = LON.max()

    # Shrink the bounding box to get a tight bounding box
    minLAT += 0.35
    maxLAT -= 0.30
    minLON += 0.15
    maxLON -= 0.40

    # idx=(LAT<maxLAT-0.30)
    # LAT2=LAT[idx]
    # print(LAT.shape, LAT2.shape)
    # idx=(LON<maxLON-0.20)
    # LON2=LON[idx]
    # print(LON.shape, LON2.shape)
    print('Map domain: ', minLAT, maxLAT, minLON, maxLON)

    # Partition the domain into ncells along each side
    hLAT = (maxLAT - minLAT) / ncell
    hLON = (maxLON - minLON) / ncell

    # Count the number of days by a hash table
    CountDays = {}
    for i in range(len(OCC_DT)):
        if OCC_DT[i] in CountDays:
            CountDays[OCC_DT[i]] += 1
        else:
            CountDays[OCC_DT[i]] = 1
    CountDays = sorted(CountDays.items(), key=operator.itemgetter(0))

    # Bucket events
    CountEvents = np.zeros((len(CountDays) * timeEachDay, 1, ncell, ncell))
    total = 0
    timeSpace = 1
    for i in range(len(CountDays)):
        for j in range(CountDays[i][1]):
            idx = (int(TIME[total][0:2]) / timeSpace) + i * timeEachDay
            idLAT = int((LAT[total] - minLAT) / hLAT)
            idLON = int((LON[total] - minLON) / hLON)
            if idLAT >= ncell:
                # print('idLAT: ', idLAT)
                idLAT = ncell - 1
            if idLON >= ncell:
                # print('idLON: ', idLON)
                idLON = ncell - 1
            CountEvents[idx, 0, idLAT, idLON] += 1
            total += 1
    print('Total number of events: ', np.sum(CountEvents))

    # Bucket daily cumulated number of events in each cell
    cumCountEvents = np.zeros((len(CountDays) * timeEachDay, 1, ncell, ncell))
    tmpsum = np.zeros((ncell, ncell))
    for i in range(len(CountDays) * timeEachDay):
        if i % timeEachDay == 0:
            tmpsum = CountEvents[i, 0, :, :]
        else:
            tmpsum = tmpsum + CountEvents[i, 0, :, :]
        cumCountEvents[i, :, :] = tmpsum
    print('Total number of events2: ', np.sum(cumCountEvents[23:cumCountEvents.shape[0]:24, 0, :, :]))

    num = ndays * timeEachDay

    #    dataall1=cumCountEvents[-num:]
    #    print('dataall shape: ', dataall1.shape)
    #    out=open('data_CrimeLA_2.csv', 'w')
    #    for i in range(dataall1.shape[0]):
    #	#for j in range(dataall1.shape[2]):
    #	#    for k in range(dataall1.shape[3]):
    #	for j in range(9, 25):
    #	    for k in range(11, 27):
    #		out.write('%d, %d, %d, %d'%(i+1, j+1, k+1, dataall1[i, :, j, k]))
    #		out.write("\n")
    #    out.close()
    return CountEvents[-num:, :, 9:25, 11:27], cumCountEvents[-num:, :, 9:25, 11:27], len(CountDays)


# Generate feature set
def create_dataset(Events=None, len_closeness=3, len_period=3, len_trend=3, TrendInterval=7, PeriodInterval=1,
                   timeEachDay=24, ndays=200):
    XC = []
    XP = []
    XT = []
    Y = []
    idx = []
    # For current image, the index of the dependent image
    # depends=[range(1, len_closeness+1), [PeriodInterval*timeEachDay*j for j in range(1, len_period+1)], [TrendInterval*timeEachDay*j for j in range(1, len_trend+1)]]
    depends = [range(2, len_closeness + 2), [PeriodInterval * timeEachDay * j for j in range(1, len_period + 1)],
               [TrendInterval * timeEachDay * j for j in range(1, len_trend + 1)]]
    i = max(TrendInterval * len_trend * timeEachDay, PeriodInterval * len_period * timeEachDay, len_closeness)
    for j in range(i, timeEachDay * ndays):
        x_c = [Events[j - k, :, :, :] for k in depends[0]]
        x_p = [Events[j - k, :, :, :] for k in depends[1]]
        x_t = [Events[j - k, :, :, :] for k in depends[2]]
        y = Events[j, :, :, :]
        idx.append(j)

        if len_closeness > 0:
            XC.append(np.vstack(x_c))
        if len_period > 0:
            XP.append(np.vstack(x_p))
        if len_trend > 0:
            XT.append(np.vstack(x_t))
        Y.append(y)
    XC = np.asarray(XC)
    XP = np.asarray(XP)
    XT = np.asarray(XT)
    Y = np.asarray(Y)
    idx = np.asarray(idx)
    # XC shape: (*, 3, 32, 32) XP shape: (*, 3, 32, 32) XT shape: (*, 3, 32, 32) Y shape: (*, 2, 32, 32)
    # idx shape: (*, )
    # * = 9096
    print('XC shape: ', XC.shape, 'XP shape: ', XP.shape, 'XT shape: ', XT.shape, 'Y shape: ', Y.shape, 'idx shape: ',
          idx.shape)
    return XC, XP, XT, Y, idx


# Generate training and testing sets
def FeatureGeneration(len_closeness=3, len_period=3, len_trend=3, TrendInterval=7, PeriodInterval=1, len_test=336,
                      meteorol_data=True, holiday_data=True, meta_data=True, ncell=30, ndays=200, timeEachDay=24,
                      filename='Crime2014_2015.csv', scale=2.0):
    assert (len_closeness + len_period + len_trend > 0)
    timeSpace = timeEachDay / 24

    # Generate data
    CountEvents, cumCountEvents, numDays = BucketData(filename=filename, ncell=ncell, ndays=ndays,
                                                      timeEachDay=timeEachDay)
    # Enlarge the data by cubic spline
    print('cumCountEvents: ', cumCountEvents.shape)

    ##Spatial Super-resolution
    # cumCountEventstmp=np.zeros((cumCountEvents.shape[0], 1, int(cumCountEvents.shape[2]*scale), int(cumCountEvents.shape[3]*scale)))
    # for i in range(cumCountEvents.shape[0]):
    #	cumCountEventstmp[i, 0, :, :]=img_enlarge(cumCountEvents[i, 0, :, :], scale, 2)
    # cumCountEvents=np.zeros((cumCountEventstmp.shape[0], 1, cumCountEventstmp.shape[1], cumCountEventstmp.shape[2]))
    # cumCountEvents=cumCountEventstmp
    # print('cumCountEvents spatial super resolution: ', cumCountEvents.shape)

    # Spatial temporal super resolution (Comment previous )
    cumCountEventstmp = img_enlarge(cumCountEvents[:, 0, :, :], scale, 2)
    cumCountEvents = np.zeros((cumCountEventstmp.shape[0], 1, cumCountEventstmp.shape[1], cumCountEventstmp.shape[2]))
    cumCountEvents[:, 0, :, :] = cumCountEventstmp
    print('cumCountEvents spatial temporal super resolution: ', cumCountEvents.shape)

    # Normalization of the data
    data_train = cumCountEvents[:-len_test]
    mmn = MinMaxNormalization()
    mmn.fit(data_train)
    data_all_mmn = [mmn.transform(d) for d in cumCountEvents]
    data_all_mmn = np.asarray(data_all_mmn)
    fpkl = open('preprocessing.pkl', 'wb')
    for obj in [mmn]:
        pickle.dump(obj, fpkl)
    fpkl.close()

    # imgplot=plt.imshow(data_all_mmn[12, 0, :, :])
    # plt.colorbar()
    # plt.show()

    # Create training and testing set. XC, XP, XT: training; Y: testing
    XC, XP, XT, Y, idx = create_dataset(Events=data_all_mmn, len_closeness=len_closeness, len_period=len_period,
                                        len_trend=len_trend, TrendInterval=TrendInterval, PeriodInterval=PeriodInterval,
                                        timeEachDay=timeEachDay, ndays=ndays)

    # Training and test
    XC_train, XP_train, XT_train, Y_train = XC[:-len_test], XP[:-len_test], XT[:-len_test], Y[:-len_test]
    XC_test, XP_test, XT_test, Y_test = XC[-len_test:], XP[-len_test:], XT[-len_test:], Y[-len_test:]

    # External features
    meta_feature = []
    if meta_data:
        Time = GenerateTimeFeatures(timeEachDay, numDays, ndays=ndays)
        print('Time shape: ', Time.shape)
    if holiday_data:
        Holiday = load_holiday(timeEachDay, numDays, ndays=ndays, fname="Holiday.txt")
        print('Holiday shape: ', Holiday.shape)
    if meteorol_data:
        Weather = load_meteorol(timeEachDay, numDays, ndays=ndays, fname='WeatherFeature.csv')
        print('Weather shape: ', Weather.shape)

    metadata_dim = 0
    if len(Time.shape) == 1:
        metadata_dim += 1
    else:
        metadata_dim += Time.shape[1]
    if len(Holiday.shape) == 1:
        metadata_dim += 1
    else:
        metadata_dim += Holiday.shape[1]
    if len(Weather.shape) == 1:
        metadata_dim += 1
    else:
        metadata_dim += Weather.shape[1]

    meta_feature = np.zeros((len(Holiday), metadata_dim))
    for i in range(len(Holiday)):
        meta_feature[i, 0] = Time[i]
        meta_feature[i, 1] = Holiday[i]
        meta_feature[i, 2:] = Weather[i, :]
    meta_feature = meta_feature[idx, :]
    print('meta_feature shape: ', meta_feature.shape)
    meta_feature_train, meta_feature_test = meta_feature[:-len_test, :], meta_feature[-len_test:, :]

    X_train = []
    X_test = []
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_train, XP_train, XT_train]):
        if l > 0:
            X_train.append(X_)
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_test, XP_test, XT_test]):
        if l > 0:
            X_test.append(X_)
    X_train.append(meta_feature_train)
    X_test.append(meta_feature_test)

    # (*, 3, 40, 40), (*, 3, 40, 40), (*, 3, 40, 40), (*, 5)
    for _X in X_train:
        print(_X.shape, )
    # (*, 3, 40, 40), (*, 3, 40, 40), (*, 3, 40, 40), (*, 5)
    for _X in X_test:
        print(_X.shape, )

    idx_train = idx[:-len_test]
    idx_test = idx[-len_test:]
    return X_train, Y_train, X_test, Y_test, mmn, metadata_dim, idx_train, idx_test


# The main function
if __name__ == '__main__':
    #    #Test enlarge image
    #    img0=np.zeros((16, 16))
    #    for i in range(16):
    #	for j in range(16):
    #	    img0[i, j]=float((i-7)*(j-7))/float(8*8)
    #    img0[0, :]=0; img0[-1, :]=0
    #    img0[:, 0]=0; img0[:, -1]=0
    #    #Cubic spline interpolation
    #    img1=img_enlarge(img0, 4.0, 2)
    #    img2=img_enlarge(img1, 0.25, 2)
    #    res=img0-img2
    #    #imgplot=plt.imshow(res)
    #    #plt.colorbar()
    #    #plt.show()
    #
    #    CountEvents, cumCountEvents, CountDays=BucketData(filename='Crime2014_2015.csv', ncell=30, ndays=60, timeEachDay=24)
    #
    #    #TODO: enlarge image to get a smoother representation
    #    cumCountEvents2=img_enlarge(cumCountEvents, factor=2.5, order=2)
    #    for i in range(cumCountEvents2.shape[0]):
    #	imgplot=plt.imshow(cumCountEvents2[i, 0, :, :])
    #	plt.pause(0.05)
    #    plt.colorbar()
    #    plt.show()
    FeatureGeneration(len_closeness=3, len_period=3, len_trend=3, len_test=336, meteorol_data=True, holiday_data=True,
                      meta_data=True, ncell=30, ndays=200, timeEachDay=24, filename='Crime2014_2015.csv', scale=2.0)
