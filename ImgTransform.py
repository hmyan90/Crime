#!/usr/bin/python
'''
This code implements two function: enlarge image by bilinear interpolation; and
shrinke image
'''
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# order=1: bilinear interpolation
# order=2: cubic interpolation
# factor: enlarged ratio
def img_enlarge(img, factor=2, order=1):
    return scipy.ndimage.zoom(img, factor, order=order)


if __name__ == "__main__":
    img0 = np.zeros((16, 16))
    for i in range(16):
        for j in range(16):
            img0[i, j] = float((i - 7) * (j - 7)) / float(8 * 8)

    img = np.zeros((10, 1, 16, 16))
    for i in range(10):
        img[i, 0, :, :] = img0 + i
    print('img shape: ', img.shape)

    img2 = np.zeros((10, 1, 8, 8))
    for i in range(10):
        img2[i, 0, :, :] = img_enlarge(img[i, 0, :, :], 0.5, 3)

    #    img0[0, :]=0; img0[-1, :]=0
    #    img0[:, 0]=0; img0[:, -1]=0
    #    imgplot=plt.imshow(img0)
    #    plt.colorbar()
    #    plt.show()
    #
    # Cubic spline interpolation
    img1 = img_enlarge(img0, 4.0, 3)
    print(img0.shape, img1.shape)
    imgplot = plt.imshow(img1)
    plt.colorbar()
    plt.show()

# img2=img_enlarge(img1, 0.25, 3)
#    res=img0-img2
#    imgplot=plt.imshow(res)
#    plt.colorbar()
#    plt.show()
#
#    #Test multiple image (:, :, :), iteratively registration
