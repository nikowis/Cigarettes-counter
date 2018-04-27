import cv2 as cv
import numpy as np

img = imgbin = cv.imread('./../RecordedImage_GO-5000M-PGE_00-0C-DF-09-1B-B4_000.tif', cv.IMREAD_GRAYSCALE)
imgbin = cv.GaussianBlur(imgbin, (3, 3), 1)
cv.imwrite('./../blur.png', imgbin)
imgbin = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 201, 2)
cv.imwrite('./../blurblack.png', imgbin)
circles = cv.HoughCircles(imgbin, cv.HOUGH_GRADIENT, 5, 20, param1=50, param2=48, minRadius=12, maxRadius=17)
print('Found ', circles[0, :,0 ].shape[0], 'circles')
cv.imwrite('circles.png', circles)
if circles is not None:
    circles = np.uint16(np.around(circles))

    for i in circles[0, :]:
        # draw the outer circle
        cv.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

cv.imwrite('./../circles.png', img)


print('Exiting')
