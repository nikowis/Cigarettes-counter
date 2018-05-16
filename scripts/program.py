import cv2 as cv
import numpy as np

img = imgbin = cv.imread('./../RecordedImage_GO-5000M-PGE_00-0C-DF-09-1B-B4_003.tif', cv.IMREAD_GRAYSCALE)
imgblur = cv.GaussianBlur(imgbin, (3, 3), 1)


ret1, dark_reduct = cv.threshold(imgblur, 90, 255, cv.THRESH_BINARY)

for i in range(0, dark_reduct.shape[0]):
    for j in range(0, dark_reduct.shape[1]):
        if dark_reduct[i][j] == 0:
            imgblur[i][j] = 0

cv.imwrite('./../blur.png', imgblur)

imgblurblack = cv.adaptiveThreshold(imgblur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 201, 2)
cv.imwrite('./../blurblack.png', imgblurblack)

edges = cv.Canny(imgblurblack, 0, 50)
cv.imwrite('./../edges.png', edges)
circles = cv.HoughCircles(imgblurblack, cv.HOUGH_GRADIENT, 5, 20, param1=50, param2=48, minRadius=12, maxRadius=17)
print('Found ', circles[0, :, 0].shape[0], 'circles')
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
