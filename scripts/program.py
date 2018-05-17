import cv2 as cv
import numpy as np

img = imgbin = cv.imread('./../RecordedImage_GO-5000M-PGE_00-0C-DF-09-1B-B4_001.tif', cv.IMREAD_GRAYSCALE)
imgblur = cv.GaussianBlur(imgbin, (3, 3), 1)

ret1, dark_reduct = cv.threshold(imgblur, 90, 255, cv.THRESH_BINARY)

imgblur[dark_reduct < 1] = 0

imgblurblack = cv.adaptiveThreshold(imgblur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 121, 2)
cv.imwrite('./../blurblack.png', imgblurblack)

edges = cv.Canny(imgblurblack, 0, 50)
cv.imwrite('./../edges.png', edges)
circles = cv.HoughCircles(imgblurblack, cv.HOUGH_GRADIENT, 5, 22, param1=50, param2=44, minRadius=12, maxRadius=17)
print('Hough found ', circles[0, :, 0].shape[0], 'circles')

img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
real_circles = 0
if circles is not None:
    circles = np.uint16(np.around(circles))

    for i in circles[0, :]:
        if imgblurblack[i[1], i[0]] != 0 and imgblur[i[1], i[0]]>100 :
            # draw the outer circle
            cv.circle(img, (i[0], i[1]), i[2], (0, 0, 0), 2)
            # draw the center of the circle
            cv.circle(img, (i[0], i[1]), 2, (37, 37, 237), 2)
            real_circles += 1

print('Really found ', real_circles, ' circles')


cv.imwrite('./../circles.png', img)

print('Exiting')
