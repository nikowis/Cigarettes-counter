import cv2 as cv
import numpy as np
import time

INPUT_FILE_PATH = './../resources/1.tif'
SAVE_EACH_PROCESS_STEP = False
start_time = time.time()

img = imgbin = cv.imread(INPUT_FILE_PATH, cv.IMREAD_GRAYSCALE)
imgblur = cv.GaussianBlur(imgbin, (3, 3), 1)

if SAVE_EACH_PROCESS_STEP:
    cv.imwrite('./step1_blur.png', imgblur)

ret1, dark_reduct = cv.threshold(imgblur, 90, 255, cv.THRESH_BINARY)
imgblur[dark_reduct < 1] = 0
if SAVE_EACH_PROCESS_STEP:
    cv.imwrite('./step2_dark_reduct.png', dark_reduct)
    cv.imwrite('./step3_mask_reduct.png', imgblur)

imgblurblack = cv.adaptiveThreshold(imgblur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 121, 2)
if SAVE_EACH_PROCESS_STEP:
    cv.imwrite('./step4_binarize.png', imgblurblack)

edges = cv.Canny(imgblurblack, 0, 50)
if SAVE_EACH_PROCESS_STEP:
    cv.imwrite('./step5_edges.png', edges)
circles = cv.HoughCircles(imgblurblack, cv.HOUGH_GRADIENT, 5, 22, param1=50, param2=38, minRadius=12, maxRadius=17)

img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
real_circles = 0
if circles is not None:
    circles = np.uint16(np.around(circles))

    for i in circles[0, :]:
        if imgblurblack[i[1], i[0]] != 0 and imgblur[i[1], i[0]] > 100:
            # draw the outer circle
            cv.circle(img, (i[0], i[1]), i[2], (0, 0, 0), 2)
            # draw the center of the circle
            cv.circle(img, (i[0], i[1]), 2, (37, 37, 237), 2)
            real_circles += 1

print('Found ', real_circles, ' circles')
print("Time {0} seconds".format((time.time() - start_time)))

cv.imwrite('./circles.png', img)

print('Exiting')
